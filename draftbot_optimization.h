//
// Created by Devon Richards on 9/3/2020.
//
#ifndef DRAFTBOTOPTIMIZATION_DRAFTBOT_OPTIMIZATION_H
#define DRAFTBOTOPTIMIZATION_DRAFTBOT_OPTIMIZATION_H
#include <array>
#include <cstddef>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

// Algorithmic Hyperparameters
constexpr float MAX_SCORE = 10.f;
constexpr float MIN_WEIGHT = 0.f;
constexpr float MAX_WEIGHT = 10.f;

// Output Parameters
constexpr size_t WIDTH = 14;
constexpr size_t PRECISION = 10;

// Optimization Hyperparameters
constexpr float FRACTION_OF_WORK_GROUPS = 0.25f;
constexpr size_t POPULATION_SIZE = 16;
constexpr size_t KEEP_BEST = 8;
constexpr size_t PICKS_PER_GENERATION = 32*1024;
constexpr double CATEGORICAL_CROSSENTROPY_LOSS_WEIGHT = 0.0;
constexpr double NEGATIVE_LOG_ACCURACY_LOSS_WEIGHT = 1.0;

// Architectural Parameters
constexpr size_t NUM_CARDS = 21467;
constexpr size_t NUM_PICK_FILES = 6;
constexpr size_t PACKS = 3;
constexpr size_t PACK_SIZE = 15;
constexpr size_t EMBEDDING_SIZE = 64;
constexpr size_t NUM_COLORS = 5;
constexpr size_t MAX_PACK_SIZE = 20;
constexpr size_t MAX_SEEN = 408;
constexpr size_t MAX_PICKED = 64;
constexpr size_t NUM_COMBINATIONS = 32;
constexpr size_t PROB_DIM_0_EXP = 4;
constexpr size_t PROB_DIM_1_EXP = 4;
constexpr size_t PROB_DIM_2_EXP = 2;
constexpr size_t PROB_DIM_3_EXP = 5;
constexpr size_t PROB_DIM_4_EXP = 5;
constexpr size_t PROB_DIM_5_EXP = 5;
constexpr size_t PROB_DIM_0 = 1 << PROB_DIM_0_EXP;
constexpr size_t PROB_DIM_1 = 1 << PROB_DIM_1_EXP;
constexpr size_t PROB_DIM_2 = 1 << PROB_DIM_2_EXP;
constexpr size_t PROB_DIM_3 = 1 << PROB_DIM_3_EXP;
constexpr size_t PROB_DIM_4 = 1 << PROB_DIM_4_EXP;
constexpr size_t PROB_DIM_5 = 1 << PROB_DIM_5_EXP;
constexpr size_t PROB_DIM = PROB_DIM_0 * PROB_DIM_1 * PROB_DIM_2 * PROB_DIM_3 * PROB_DIM_4 * PROB_DIM_5;
constexpr unsigned char BASIC_LAND_TYPES_REQUIRED = 2;
constexpr unsigned char LANDS_TO_INCLUDE_COLOR = 3;
constexpr std::array<char, NUM_COLORS> COLORS{'w', 'u', 'b', 'r', 'g'};
using index_type = unsigned short;
using Weights = float[PACKS][PACK_SIZE];
using Lands = unsigned char[NUM_COMBINATIONS];
constexpr size_t MAX_REQUIREMENTS = 5;
struct ColorRequirement {
#ifdef CONSIDER_NON_BASICS
    std::pair<unsigned char[NUM_COMBINATIONS], unsigned char> requirements[5]{};
#else
    std::pair<unsigned char[8], unsigned char> requirements[5]{};
#endif
    unsigned char requirements_count{0};
    size_t offset{0};

    ColorRequirement() = default;
    ColorRequirement(const unsigned char requirements_count, const size_t offset) : requirements_count(requirements_count), offset(offset) {}
    constexpr ColorRequirement(const ColorRequirement& other) {
        for (size_t i=0; i < 5; i++) {
#ifdef CONSIDER_NON_BASICS
            for (size_t j=0; j < NUM_COMBINATIONS; j++) requirements[i].first[j] = other.requirements[i].first[j];
#else
            for (size_t j=0; j < 8; j++) requirements[i].first[j] = other.requirements[i].first[j];
#endif
            requirements[i].second = other.requirements[i].second;
        }
        requirements_count = other.requirements_count;
        offset = other.offset;
    }
    constexpr ColorRequirement& operator=(const ColorRequirement& other) {
        for (size_t i=0; i < 5; i++) {
#ifdef CONSIDER_NON_BASICS
            for (size_t j=0; j < NUM_COMBINATIONS; j++) requirements[i].first[j] = other.requirements[i].first[j];
#else
            for (size_t j=0; j < 8; j++) requirements[i].first[j] = other.requirements[i].first[j];
#endif
            requirements[i].second = other.requirements[i].second;
        }
        requirements_count = other.requirements_count;
        offset = other.offset;
        return *this;
    }
};
using Embedding = std::array<float, EMBEDDING_SIZE>;
//using CastingProbabilityTable = std::array<std::array<std::array<std::array<std::array<std::array<float, PROB_DIM_5>, PROB_DIM_4>, PROB_DIM_3>, PROB_DIM_2>, PROB_DIM_1>, PROB_DIM_0>;

constexpr float INITIAL_IS_FETCH_MULTIPLIER = 1.f;
constexpr float INITIAL_HAS_BASIC_TYPES_MULTIPLIER = 0.75f;
constexpr float INITIAL_IS_REGULAR_LAND_MULTIPLIER = 0.5f;
constexpr float INITIAL_EQUAL_CARDS_SYNERGY = 2.5f;
constexpr Weights INITIAL_RATING_WEIGHTS{
    {5 / 6.f, 5 / 6.f, 5 / 6.f, 5 / 6.f, 5 / 6.f, 5 / 6.f, 5 / 6.f, 5 / 6.f, 5 / 6.f, 5 / 6.f, 5 / 6.f, 5 / 6.f, 5 / 6.f, 5 / 6.f, 5 / 6.f},
    {4 / 6.f, 4 / 6.f, 4 / 6.f, 4 / 6.f, 4 / 6.f, 4 / 6.f, 4 / 6.f, 4 / 6.f, 4 / 6.f, 4 / 6.f, 4 / 6.f, 4 / 6.f, 4 / 6.f, 4 / 6.f, 4 / 6.f},
    {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f},
};
constexpr Weights INITIAL_COLORS_WEIGHTS{
    {20 / 6.f, 20 / 6.f, 20 / 6.f, 20 / 6.f, 20 / 6.f, 20 / 6.f, 20 / 6.f, 20 / 6.f, 20 / 6.f, 20 / 6.f, 20 / 6.f, 20 / 6.f, 20 / 6.f, 20 / 6.f, 20 / 6.f},
    {40 / 6.f, 40 / 6.f, 40 / 6.f, 40 / 6.f, 40 / 6.f, 40 / 6.f, 40 / 6.f, 40 / 6.f, 40 / 6.f, 40 / 6.f, 40 / 6.f, 40 / 6.f, 40 / 6.f, 40 / 6.f, 40 / 6.f},
    {10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10},
};
constexpr Weights INITIAL_FIXING_WEIGHTS{
    {0.1f / 6.f, 0.3f / 6.f, 0.6f / 6.f, 0.8f / 6.f, 1 / 6.f, 1 / 6.f, 1 / 6.f, 1 / 6.f, 1 / 6.f, 1 / 6.f, 1 / 6.f, 1 / 6.f, 1 / 6.f, 1 / 6.f, 1 / 6.f},
    {1.5f / 6.f, 1.5f / 6.f, 1.5f / 6.f, 1.5f / 6.f, 1.5f / 6.f, 1.5f / 6.f, 1.5f / 6.f, 1.5f / 6.f, 1.5f / 6.f, 1.5f / 6.f, 1.5f / 6.f, 1.5f / 6.f, 1.5f / 6.f, 1.5f / 6.f, 1.5f / 6.f},
    {1 / 6.f, 1 / 6.f, 1 / 6.f, 1 / 6.f, 1 / 6.f, 1 / 6.f, 1 / 6.f, 1 / 6.f, 1 / 6.f, 1 / 6.f, 1 / 6.f, 1 / 6.f, 1 / 6.f, 1 / 6.f, 1 / 6.f},
};
constexpr Weights INITIAL_INTERNAL_SYNERGY_WEIGHTS{
    {3 / 6.f, 3 / 6.f, 3 / 6.f, 3 / 6.f, 3 / 6.f, 3 / 6.f, 3 / 6.f, 3 / 6.f, 3 / 6.f, 3 / 6.f, 3 / 6.f, 3 / 6.f, 3 / 6.f, 3 / 6.f, 3 / 6.f},
    {4 / 6.f, 4 / 6.f, 4 / 6.f, 4 / 6.f, 4 / 6.f, 4 / 6.f, 4 / 6.f, 4 / 6.f, 4 / 6.f, 4 / 6.f, 4 / 6.f, 4 / 6.f, 4 / 6.f, 4 / 6.f, 4 / 6.f},
    {5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5},
};
constexpr Weights INITIAL_PICK_SYNERGY_WEIGHTS{
    {3 / 6.f, 3 / 6.f, 3 / 6.f, 3 / 6.f, 3 / 6.f, 3 / 6.f, 3 / 6.f, 3 / 6.f, 3 / 6.f, 3 / 6.f, 3 / 6.f, 3 / 6.f, 3 / 6.f, 3 / 6.f, 3 / 6.f},
    {4 / 6.f, 4 / 6.f, 4 / 6.f, 4 / 6.f, 4 / 6.f, 4 / 6.f, 4 / 6.f, 4 / 6.f, 4 / 6.f, 4 / 6.f, 4 / 6.f, 4 / 6.f, 4 / 6.f, 4 / 6.f, 4 / 6.f},
    {5 / 6.f, 5 / 6.f, 5 / 6.f, 5 / 6.f, 5 / 6.f, 5 / 6.f, 5 / 6.f, 5 / 6.f, 5 / 6.f, 5 / 6.f, 5 / 6.f, 5 / 6.f, 5 / 6.f, 5 / 6.f, 5 / 6.f},
};
constexpr Weights INITIAL_OPENNESS_WEIGHTS{
    {4 / 6.f, 12 / 6.f, 12.3f / 6.f, 12.6f / 6.f, 13 / 6.f, 13.4f / 6.f, 13.7f / 6.f, 14 / 6.f, 15 / 6.f, 14.6f / 6.f, 14.2f / 6.f, 13.8f / 6.f, 13.4f / 6.f, 13 / 6.f, 12.6f / 6.f},
    {13 / 6.f, 12.6f / 6.f, 12.2f / 6.f, 11.8f / 6.f, 11.4f / 6.f, 11 / 6.f, 10.6f / 6.f, 10.2f / 6.f, 9.8f / 6.f, 9.4f / 6.f, 9 / 6.f, 8.6f / 6.f, 8.2f / 6.f, 7.8f / 6.f, 7 / 6.f},
    {8 / 6.f, 7.5f / 6.f, 7 / 6.f, 6.5f / 6.f, 1, 5.5f / 6.f, 5 / 6.f, 4.5f / 6.f, 4 / 6.f, 3.5f / 6.f, 3 / 6.f, 2.5f / 6.f, 2 / 6.f, 1.5f / 6.f, 1 / 6.f},
};
constexpr float INITIAL_PROB_TO_INCLUDE = 0.4f;
constexpr float INITIAL_SIMILARITY_CLIP = 0.7f;
constexpr Lands DEFAULT_LANDS{0, 4, 4, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
constexpr std::array<std::array<bool, NUM_COLORS>, NUM_COMBINATIONS> COLOR_COMBINATIONS{{
    {false, false, false, false, false},
    {true, false, false, false, false}, // 1
    {false, true, false, false, false},
    {false, false, true, false, false},
    {false, false, false, true, false},
    {false, false, false, false, true},
    {true, true, false, false, false}, // 6
    {false, true, true, false, false},
    {false, false, true, true, false},
    {false, false, false, true, true},
    {true, false, false, false, true},
    {true, false, true, false, false}, // 11
    {false, true, false, true, false},
    {false, false, true, false, true},
    {true, false, false, true, false},
    {false, true, false, false, true},
    {true, true, false, false, true}, // 16
    {true, true, true, false, false},
    {false, true, true, true, false},
    {false, false, true, true, true},
    {true, false, false, true, true},
    {true, false, true, true, false}, // 21
    {false, true, false, true, true},
    {true, false, true, false, true},
    {true, true, false, true, false},
    {false, true, true, false, true},
    {false, true, true, true, true}, // 26
    {true, false, true, true, true},
    {true, true, false, true, true},
    {true, true, true, false, true},
    {true, true, true, true, false},
    {true, true, true, true, true}, // 31
}};
constexpr std::array<std::array<size_t, 16>, NUM_COLORS> INCLUSION_MAP{{
    {1, 6,10, 11, 14, 16, 17, 20, 21, 23, 24, 27, 28, 29, 30, 31},
    {2, 6, 7, 12, 15, 16, 17, 18, 22, 24, 25, 26, 28, 29, 30, 31},
    {3, 7, 8, 11, 13, 17, 18, 19, 21, 23, 25, 26, 27, 29, 30, 31},
    {4, 8, 9, 12, 14, 18, 19, 20, 21, 22, 24, 26, 27, 28, 30, 31},
    {5, 9,10, 13, 15, 16, 19, 20, 22, 23, 25, 26, 27, 28, 29, 31},
}};
extern std::array<float, NUM_CARDS> INITIAL_RATINGS;

template <typename Generator>
void generate_weights(Generator& gen, Weights& result) {
    std::uniform_real_distribution<float> weight_dist(MIN_WEIGHT, MAX_WEIGHT);
    for (size_t i=0; i < PACKS; i++) {
        for (size_t j=0; j < PACK_SIZE; j++) result[i][j] = weight_dist(gen);
    }
}

constexpr float sigmoid_temp = MAX_SCORE / 5;

template<typename Scalar>
Scalar sigmoid(const Scalar value, const Scalar max, const Scalar min=0) {
    Scalar exp = std::exp(value / sigmoid_temp);
    return (max - min) * exp / (1 + exp) + min;
}

template <typename Scalar>
Scalar inverse_sigmoid(const Scalar value, const Scalar max, const Scalar min=0) {
    if (value - min >= max) return static_cast<Scalar>(6 * sigmoid_temp);
    else if (value - min <= 0) return static_cast<Scalar>(-6 * sigmoid_temp);
    return -std::log((max - value) / (value - min)) * sigmoid_temp;
}

constexpr size_t WEIGHT_PARAMETER_COUNT = PACK_SIZE * PACKS;
template <size_t Size>
void array_to_weights(const std::array<float, Size>& params, size_t start_index, Weights& result) {
    for (size_t i=0; i < PACKS; i++) {
        for (size_t j=0; j < PACK_SIZE; j++) result[i][j] = sigmoid(params[start_index + i * PACK_SIZE + j], MAX_WEIGHT, MIN_WEIGHT);
    }
}

template <size_t Size>
void array_to_weights(const std::array<float, Size>& params, size_t start_index, Weights& result, bool) {
    for (size_t i = 0; i < PACKS; i++) {
        for (size_t j = 0; j < PACK_SIZE; j++) result[i][j] = params[start_index + i * PACK_SIZE + j];
    }
}

template <size_t Size>
void write_weights_to_array(const Weights& weights, std::array<float, Size>& params, size_t start_index) {
    for (size_t i = 0; i < PACKS; i++) {
        for (size_t j=0; j < PACK_SIZE; j++) params[start_index + i * PACK_SIZE + j] = inverse_sigmoid(weights[i][j], MAX_WEIGHT, MIN_WEIGHT);
    }
}

struct Variables {
    Weights rating_weights;
    Weights colors_weights;
    Weights fixing_weights;
    Weights internal_synergy_weights;
    Weights pick_synergy_weights;
    Weights openness_weights;
#ifdef OPTIMIZE_RATINGS
    float ratings[NUM_CARDS]{1};
    static constexpr size_t num_parameters = 6 * WEIGHT_PARAMETER_COUNT + NUM_CARDS + 6;
#else
    static constexpr size_t num_parameters = 6 * WEIGHT_PARAMETER_COUNT + 6;
#endif
    float prob_to_include = INITIAL_PROB_TO_INCLUDE;
    float prob_multiplier = 1 / (1 - INITIAL_PROB_TO_INCLUDE);
    float similarity_clip = INITIAL_SIMILARITY_CLIP;
    float similarity_multiplier = 1 / (1 - INITIAL_SIMILARITY_CLIP);
    float is_fetch_multiplier = INITIAL_IS_FETCH_MULTIPLIER;
    float has_basic_types_multiplier = INITIAL_HAS_BASIC_TYPES_MULTIPLIER;
    float is_regular_land_multiplier = INITIAL_IS_REGULAR_LAND_MULTIPLIER;
    float equal_cards_synergy = INITIAL_EQUAL_CARDS_SYNERGY;

    Variables() {
        for (size_t i=0; i < PACKS; i++) for (size_t j=0; j < PACK_SIZE; j++) rating_weights[i][j] = INITIAL_RATING_WEIGHTS[i][j];
        for (size_t i=0; i < PACKS; i++) for (size_t j=0; j < PACK_SIZE; j++) pick_synergy_weights[i][j] = INITIAL_PICK_SYNERGY_WEIGHTS[i][j];
        for (size_t i=0; i < PACKS; i++) for (size_t j=0; j < PACK_SIZE; j++) fixing_weights[i][j] = INITIAL_FIXING_WEIGHTS[i][j];
        for (size_t i=0; i < PACKS; i++) for (size_t j=0; j < PACK_SIZE; j++) internal_synergy_weights[i][j] = INITIAL_INTERNAL_SYNERGY_WEIGHTS[i][j];
        for (size_t i=0; i < PACKS; i++) for (size_t j=0; j < PACK_SIZE; j++) openness_weights[i][j] = INITIAL_OPENNESS_WEIGHTS[i][j];
        for (size_t i=0; i < PACKS; i++) for (size_t j=0; j < PACK_SIZE; j++) colors_weights[i][j] = INITIAL_COLORS_WEIGHTS[i][j];
#ifdef OPTIMIZE_RATINGS
        for (size_t i=0; i < NUM_CARDS; i++) ratings[i] = INITIAL_RATINGS[i];
#endif
    }

    explicit Variables(std::mt19937_64 & gen) {
        std::uniform_real_distribution<float> rating_dist{0.f, MAX_SCORE};
        std::uniform_real_distribution<float> unit_dist{0.f, 1.f};
        generate_weights(gen, rating_weights);
        generate_weights(gen, pick_synergy_weights);
        generate_weights(gen, fixing_weights);
        generate_weights(gen, internal_synergy_weights);
        generate_weights(gen, openness_weights);
        generate_weights(gen, colors_weights);
#ifdef OPTIMIZE_RATINGS
        for (size_t i=0; i < NUM_CARDS; i++) ratings[i] = rating_dist(gen);
#endif
        prob_to_include = std::min(unit_dist(gen), 0.99f);
        prob_multiplier = 1 / (1 - prob_to_include);
        similarity_clip = std::min(unit_dist(gen), 0.99f);
        similarity_multiplier = 1 / (1 - similarity_clip);
        is_fetch_multiplier = unit_dist(gen);
        has_basic_types_multiplier = unit_dist(gen);
        is_regular_land_multiplier = unit_dist(gen);
        equal_cards_synergy = rating_dist(gen);
    }

    explicit Variables(const std::array<float, num_parameters>& params) {
        array_to_weights(params, 0, rating_weights);
        array_to_weights(params, WEIGHT_PARAMETER_COUNT, pick_synergy_weights);
        array_to_weights(params, 2 * WEIGHT_PARAMETER_COUNT, fixing_weights);
        array_to_weights(params, 3 * WEIGHT_PARAMETER_COUNT, internal_synergy_weights);
        array_to_weights(params, 4 * WEIGHT_PARAMETER_COUNT, openness_weights);
        array_to_weights(params, 5 * WEIGHT_PARAMETER_COUNT, colors_weights);
#ifdef OPTIMIZE_RATINGS
        for (size_t i=0; i < NUM_CARDS; i++) ratings[i] = sigmoid(params[6 * WEIGHT_PARAMETER_COUNT + i], MAX_SCORE);
        constexpr size_t start_index = 6 * WEIGHT_PARAMETER_COUNT + NUM_CARDS;
#else
        constexpr size_t start_index = 6 * WEIGHT_PARAMETER_COUNT;
#endif
        prob_to_include = sigmoid(params[start_index], 0.99f);
        prob_multiplier = 1 / (1 - prob_to_include);
        similarity_clip = sigmoid(params[start_index + 1], 0.99f);
        similarity_multiplier = 1 / (1 - similarity_clip);
        is_fetch_multiplier = sigmoid(params[start_index + 2], 1.f);
        has_basic_types_multiplier = sigmoid(params[start_index + 3], 1.f);
        is_regular_land_multiplier = sigmoid(params[start_index + 4], 1.f);
        equal_cards_synergy = sigmoid(params[start_index + 5], MAX_SCORE);
    }

    explicit Variables(const std::array<float, num_parameters>& params, bool) {
        array_to_weights(params, 0, rating_weights, false);
        array_to_weights(params, WEIGHT_PARAMETER_COUNT, pick_synergy_weights, false);
        array_to_weights(params, 2 * WEIGHT_PARAMETER_COUNT, fixing_weights, false);
        array_to_weights(params, 3 * WEIGHT_PARAMETER_COUNT, internal_synergy_weights, false);
        array_to_weights(params, 4 * WEIGHT_PARAMETER_COUNT, openness_weights, false);
        array_to_weights(params, 5 * WEIGHT_PARAMETER_COUNT, colors_weights, false);
#ifdef OPTIMIZE_RATINGS
        for (size_t i = 0; i < NUM_CARDS; i++) ratings[i] = params[6 * WEIGHT_PARAMETER_COUNT + i];
        constexpr size_t start_index = 6 * WEIGHT_PARAMETER_COUNT + NUM_CARDS;
#else
        constexpr size_t start_index = 6 * WEIGHT_PARAMETER_COUNT;
#endif
        prob_to_include = params[start_index];
        prob_multiplier = 1 / (1 - prob_to_include);
        similarity_clip = params[start_index + 1];
        similarity_multiplier = 1 / (1 - similarity_clip);
        is_fetch_multiplier = params[start_index + 2];
        has_basic_types_multiplier = params[start_index + 3];
        is_regular_land_multiplier = params[start_index + 4];
        equal_cards_synergy = params[start_index + 5];
    }

    explicit operator std::array<float, num_parameters>() const {
        std::array<float, num_parameters> result;
        write_weights_to_array(rating_weights, result, 0);
        write_weights_to_array(pick_synergy_weights, result, WEIGHT_PARAMETER_COUNT);
        write_weights_to_array(fixing_weights, result, 2 * WEIGHT_PARAMETER_COUNT);
        write_weights_to_array(internal_synergy_weights, result, 3 * WEIGHT_PARAMETER_COUNT);
        write_weights_to_array(openness_weights, result, 4 * WEIGHT_PARAMETER_COUNT);
        write_weights_to_array(colors_weights, result, 5 * WEIGHT_PARAMETER_COUNT);
#ifdef OPTIMIZE_RATINGS
        for (size_t i=0; i < NUM_CARDS; i++) result[6 * WEIGHT_PARAMETER_COUNT + i] = inverse_sigmoid(ratings[i], MAX_SCORE);
        constexpr size_t start_index = 6 * WEIGHT_PARAMETER_COUNT + NUM_CARDS;
#else
        constexpr size_t start_index = 6 * WEIGHT_PARAMETER_COUNT;
#endif
        result[start_index + 0] = inverse_sigmoid(prob_to_include, 0.99f);
        result[start_index + 1] = inverse_sigmoid(similarity_clip, 0.99f);
        result[start_index + 2] = inverse_sigmoid(is_fetch_multiplier, 1.f);
        result[start_index + 3] = inverse_sigmoid(has_basic_types_multiplier, 1.f);
        result[start_index + 4] = inverse_sigmoid(is_regular_land_multiplier, 1.f);
        result[start_index + 5] = inverse_sigmoid(equal_cards_synergy, MAX_SCORE);
        return result;
    }
};

struct Pick {
    index_type in_pack[MAX_PACK_SIZE]{std::numeric_limits<index_type>::max()};
    index_type in_pack_count{0};
    index_type seen[MAX_SEEN]{std::numeric_limits<index_type>::max()};
    index_type seen_count{0};
    index_type picked[MAX_PICKED]{std::numeric_limits<index_type>::max()};
    index_type picked_count{0};
    unsigned char pack_num{0};
    unsigned char pick_num{0};
    unsigned char pack_size{0};
    unsigned char packs{0};
    index_type chosen_card{0};
};
extern const std::map<std::string, std::array<bool, 5>> FETCH_LANDS;

struct Constants {
    ColorRequirement color_requirements[NUM_CARDS]; // NOLINT(cert-err58-cpp)
    unsigned char cmcs[NUM_CARDS];
    bool card_colors[NUM_CARDS][5];
    bool is_land[NUM_CARDS];
    bool is_fetch[NUM_CARDS];
    bool has_basic_land_types[NUM_CARDS];
    float similarities[NUM_CARDS][NUM_CARDS];
    float prob_to_cast[PROB_DIM];
#ifndef OPTIMIZE_RATINGS
    float ratings[NUM_CARDS];
#endif
};

constexpr float& get_prob_to_cast(float (&prob_to_cast)[PROB_DIM], const size_t cmc, const size_t required_a, const size_t land_count_a) noexcept {
    return prob_to_cast[(((cmc << PROB_DIM_1_EXP) | required_a) << (PROB_DIM_2_EXP + PROB_DIM_3_EXP + PROB_DIM_4_EXP + PROB_DIM_5_EXP)) | land_count_a];
}

constexpr float& get_prob_to_cast(float (&prob_to_cast)[PROB_DIM], const size_t cmc, const size_t required_a, const size_t required_b, const size_t land_count_a,
                                  const size_t land_count_b, const size_t land_count_ab) noexcept {
    return prob_to_cast[(((((((((cmc << PROB_DIM_1_EXP) | required_a) << PROB_DIM_2_EXP) | required_b) << PROB_DIM_3_EXP) | land_count_ab) << PROB_DIM_4_EXP) | land_count_b) << PROB_DIM_5_EXP) | land_count_a];
}

constexpr float get_prob_to_cast(const float *prob_to_cast, const size_t cmc, const size_t required_a, const size_t land_count_a) noexcept {
    return prob_to_cast[(((cmc << PROB_DIM_1_EXP) | required_a) << (PROB_DIM_2_EXP + PROB_DIM_3_EXP + PROB_DIM_4_EXP + PROB_DIM_5_EXP)) | land_count_a];
}

constexpr float get_prob_to_cast(const float *prob_to_cast, const size_t cmc, const size_t required_a, const size_t required_b, const size_t land_count_a,
                                 const size_t land_count_b, const size_t land_count_ab) noexcept {
    return prob_to_cast[(((((((((cmc << PROB_DIM_1_EXP) | required_a) << PROB_DIM_2_EXP) | required_b) << PROB_DIM_3_EXP) | land_count_ab) << PROB_DIM_4_EXP) | land_count_b) << PROB_DIM_5_EXP) | land_count_a];
}

constexpr float get_prob_to_cast(const float *prob_to_cast, const size_t offset, const size_t land_count_a) noexcept {
    return prob_to_cast[offset | land_count_a];
}

constexpr float get_prob_to_cast(const float *prob_to_cast, const size_t offset, const size_t land_count_a, const size_t land_count_b, const size_t land_count_ab) noexcept {
    return prob_to_cast[offset | (((land_count_ab << PROB_DIM_4_EXP) | land_count_b) << PROB_DIM_5_EXP) | land_count_a];
}

struct ExpandedPick : public Pick {
    float in_pack_similarities[MAX_PACK_SIZE][MAX_PICKED];
    ColorRequirement in_pack_color_requirements[MAX_PACK_SIZE];
    unsigned char in_pack_cmcs[MAX_PACK_SIZE];
    bool in_pack_is_land[MAX_PACK_SIZE];
    bool in_pack_card_colors[MAX_PACK_SIZE][5];
    bool in_pack_is_fetch[MAX_PACK_SIZE];
    bool in_pack_has_basic_land_types[MAX_PACK_SIZE];
    ColorRequirement seen_color_requirements[MAX_SEEN];
    unsigned char seen_cmcs[MAX_SEEN];
    ColorRequirement picked_color_requirements[MAX_PICKED];
    unsigned char picked_cmcs[MAX_PICKED];
    float picked_similarities[MAX_PICKED][MAX_PICKED];

    ExpandedPick(const Pick& pick, const Constants& constants) {
        *((Pick*)this) = pick;
        for (size_t i=0; i < in_pack_count; i++) {
            in_pack_color_requirements[i] = constants.color_requirements[in_pack[i]];
            in_pack_cmcs[i] = constants.cmcs[in_pack[i]];
            in_pack_is_land[i] = constants.is_land[in_pack[i]];
            in_pack_is_fetch[i] = constants.is_fetch[in_pack[i]];
            in_pack_has_basic_land_types[i] = constants.has_basic_land_types[in_pack[i]];
            for (size_t j=0; j < pick.picked_count; j++) in_pack_similarities[i][j] = constants.similarities[in_pack[i]][picked[j]];
        }
        for (size_t i=0; i < seen_count; i++) {
            seen_color_requirements[i] = constants.color_requirements[seen[i]];
            seen_cmcs[i] = constants.cmcs[seen[i]];
        }
        for (size_t i=0; i < picked_count; i++) {
            picked_color_requirements[i] = constants.color_requirements[picked[i]];
            picked_cmcs[i] = constants.cmcs[picked[i]];
            for (size_t j=0; j < picked_count; j++) picked_similarities[i][j] = constants.similarities[picked[i]][picked[j]];
        }
    }
};

void populate_constants(const std::string& file_name, Constants& constants);
void populate_prob_to_cast(const std::string& file_name, Constants& constants);
std::vector<Pick> load_picks(const std::string& folder);
void save_variables(const Variables& variables, const std::string& file_name);
Variables load_variables(const std::string& file_name);

Variables optimize_variables(float temperature, const std::vector<Pick>& picks, size_t num_generations,
                             const std::shared_ptr<const Constants>& constants,
                             const std::shared_ptr<const Variables>& initial_variables, size_t seed);

std::array<std::array<double, 4>, POPULATION_SIZE> run_simulations(const std::vector<Variables>& variables,
                                                                   const std::vector<Pick>& picks, float temperature,
                                                                   const std::shared_ptr<const Constants>& constants);
#ifdef USE_CUDA
std::array<std::array<double, 4>, POPULATION_SIZE> run_simulations_cuda(const std::vector<Variables>& variables,
                                                                        const std::vector<ExpandedPick>& picks, float temperature,
                                                                        const float (&prob_to_cast)[PROB_DIM]);
#endif
#endif //DRAFTBOTOPTIMIZATION_DRAFTBOT_OPTIMIZATION_H//
