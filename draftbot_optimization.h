//
// Created by Devon Richards on 9/3/2020.
//
#ifndef DRAFTBOTOPTIMIZATION_DRAFTBOT_OPTIMIZATION_H
#define DRAFTBOTOPTIMIZATION_DRAFTBOT_OPTIMIZATION_H
#include <array>
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
constexpr size_t POPULATION_SIZE = 32; //(size_t)((64 + 48) * FRACTION_OF_WORK_GROUPS);
constexpr size_t KEEP_BEST = 8; //(size_t)(POPULATION_SIZE * 0.65);
constexpr size_t PICKS_PER_GENERATION = 30 * 1024;
constexpr double CATEGORICAL_CROSSENTROPY_LOSS_WEIGHT = 0.25;
constexpr double NEGATIVE_LOG_ACCURACY_LOSS_WEIGHT = 5.0;

// Architectural Parameters
constexpr size_t NUM_CARDS = 21467;
constexpr size_t NUM_PICK_FILES = 1508;
constexpr size_t PACKS = 3;
constexpr size_t PACK_SIZE = 15;
constexpr size_t EMBEDDING_SIZE = 64;
constexpr size_t NUM_COLORS = 5;
constexpr size_t MAX_PACK_SIZE = 20;
constexpr size_t MAX_SEEN = 512;
constexpr size_t MAX_PICKED = 128;
constexpr size_t NUM_COMBINATIONS = 32;
constexpr size_t PROB_DIM_0 = 9;
constexpr size_t PROB_DIM_1 = 7;
constexpr size_t PROB_DIM_2 = 4;
constexpr size_t PROB_DIM_3 = 18;
constexpr size_t PROB_DIM_4 = 18;
constexpr size_t PROB_DIM_5 = 18;
constexpr unsigned char BASIC_LAND_TYPES_REQUIRED = 2;
constexpr unsigned char LANDS_TO_INCLUDE_COLOR = 3;
constexpr std::array<char, NUM_COLORS> COLORS{'w', 'u', 'b', 'r', 'g'};
using index_type = unsigned short;
using Weights = std::array<std::array<float, PACK_SIZE>, PACKS>;
using Colors = std::array<bool, NUM_COLORS>;
using Lands = std::array<std::pair<Colors, unsigned char>, NUM_COMBINATIONS>;
using ColorRequirement = std::pair<std::array<std::pair<std::array<unsigned char, NUM_COMBINATIONS>, unsigned char>, 5>, unsigned char>;
using Embedding = std::array<float, EMBEDDING_SIZE>;
using CastingProbabilityTable = std::array<std::array<std::array<std::array<std::array<std::array<float, PROB_DIM_5>, PROB_DIM_4>, PROB_DIM_3>, PROB_DIM_2>, PROB_DIM_1>, PROB_DIM_0>;

constexpr float INITIAL_IS_FETCH_MULTIPLIER = 1.f;
constexpr float INITIAL_HAS_BASIC_TYPES_MULTIPLIER = 0.75f;
constexpr float INITIAL_IS_REGULAR_LAND_MULTIPLIER = 0.5f;
constexpr float INITIAL_EQUAL_CARDS_SYNERGY = 2.5f;
constexpr Weights INITIAL_RATING_WEIGHTS{{
    {5 / 6.f, 5 / 6.f, 5 / 6.f, 5 / 6.f, 5 / 6.f, 5 / 6.f, 5 / 6.f, 5 / 6.f, 5 / 6.f, 5 / 6.f, 5 / 6.f, 5 / 6.f, 5 / 6.f, 5 / 6.f, 5 / 6.f},
    {4 / 6.f, 4 / 6.f, 4 / 6.f, 4 / 6.f, 4 / 6.f, 4 / 6.f, 4 / 6.f, 4 / 6.f, 4 / 6.f, 4 / 6.f, 4 / 6.f, 4 / 6.f, 4 / 6.f, 4 / 6.f, 4 / 6.f},
    {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f},
}};
constexpr Weights INITIAL_COLORS_WEIGHTS{{
    {20 / 6.f, 20 / 6.f, 20 / 6.f, 20 / 6.f, 20 / 6.f, 20 / 6.f, 20 / 6.f, 20 / 6.f, 20 / 6.f, 20 / 6.f, 20 / 6.f, 20 / 6.f, 20 / 6.f, 20 / 6.f, 20 / 6.f},
    {40 / 6.f, 40 / 6.f, 40 / 6.f, 40 / 6.f, 40 / 6.f, 40 / 6.f, 40 / 6.f, 40 / 6.f, 40 / 6.f, 40 / 6.f, 40 / 6.f, 40 / 6.f, 40 / 6.f, 40 / 6.f, 40 / 6.f},
    {10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10},
}};
constexpr Weights INITIAL_FIXING_WEIGHTS{{
    {0.1f / 6.f, 0.3f / 6.f, 0.6f / 6.f, 0.8f / 6.f, 1 / 6.f, 1 / 6.f, 1 / 6.f, 1 / 6.f, 1 / 6.f, 1 / 6.f, 1 / 6.f, 1 / 6.f, 1 / 6.f, 1 / 6.f, 1 / 6.f},
    {1.5f / 6.f, 1.5f / 6.f, 1.5f / 6.f, 1.5f / 6.f, 1.5f / 6.f, 1.5f / 6.f, 1.5f / 6.f, 1.5f / 6.f, 1.5f / 6.f, 1.5f / 6.f, 1.5f / 6.f, 1.5f / 6.f, 1.5f / 6.f, 1.5f / 6.f, 1.5f / 6.f},
    {1 / 6.f, 1 / 6.f, 1 / 6.f, 1 / 6.f, 1 / 6.f, 1 / 6.f, 1 / 6.f, 1 / 6.f, 1 / 6.f, 1 / 6.f, 1 / 6.f, 1 / 6.f, 1 / 6.f, 1 / 6.f, 1 / 6.f},
}};
constexpr Weights INITIAL_INTERNAL_SYNERGY_WEIGHTS{{
    {3 / 6.f, 3 / 6.f, 3 / 6.f, 3 / 6.f, 3 / 6.f, 3 / 6.f, 3 / 6.f, 3 / 6.f, 3 / 6.f, 3 / 6.f, 3 / 6.f, 3 / 6.f, 3 / 6.f, 3 / 6.f, 3 / 6.f},
    {4 / 6.f, 4 / 6.f, 4 / 6.f, 4 / 6.f, 4 / 6.f, 4 / 6.f, 4 / 6.f, 4 / 6.f, 4 / 6.f, 4 / 6.f, 4 / 6.f, 4 / 6.f, 4 / 6.f, 4 / 6.f, 4 / 6.f},
    {5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5},
}};
constexpr Weights INITIAL_PICK_SYNERGY_WEIGHTS{{
    {3 / 6.f, 3 / 6.f, 3 / 6.f, 3 / 6.f, 3 / 6.f, 3 / 6.f, 3 / 6.f, 3 / 6.f, 3 / 6.f, 3 / 6.f, 3 / 6.f, 3 / 6.f, 3 / 6.f, 3 / 6.f, 3 / 6.f},
    {4 / 6.f, 4 / 6.f, 4 / 6.f, 4 / 6.f, 4 / 6.f, 4 / 6.f, 4 / 6.f, 4 / 6.f, 4 / 6.f, 4 / 6.f, 4 / 6.f, 4 / 6.f, 4 / 6.f, 4 / 6.f, 4 / 6.f},
    {5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5},
}};
constexpr Weights INITIAL_OPENNESS_WEIGHTS{{
    {4 / 6.f, 12 / 6.f, 12.3f / 6.f, 12.6f / 6.f, 13 / 6.f, 13.4f / 6.f, 13.7f / 6.f, 14 / 6.f, 15 / 6.f, 14.6f / 6.f, 14.2f / 6.f, 13.8f / 6.f, 13.4f / 6.f, 13 / 6.f, 12.6f / 6.f},
    {13 / 6.f, 12.6f / 6.f, 12.2f / 6.f, 11.8f / 6.f, 11.4f / 6.f, 11 / 6.f, 10.6f / 6.f, 10.2f / 6.f, 9.8f / 6.f, 9.4f / 6.f, 9 / 6.f, 8.6f / 6.f, 8.2f / 6.f, 7.8f / 6.f, 7 / 6.f},
    {8 / 6.f, 7.5f / 6.f, 7 / 6.f, 6.5f / 6.f, 1, 5.5f / 6.f, 5 / 6.f, 4.5f / 6.f, 4 / 6.f, 3.5f / 6.f, 3 / 6.f, 2.5f / 6.f, 2 / 6.f, 1.5f / 6.f, 1 / 6.f},
}};
constexpr float INITIAL_PROB_TO_INCLUDE = 0.4f;
constexpr float INITIAL_SIMILARITY_CLIP = 0.7f;
constexpr Lands DEFAULT_LANDS{{
    {{false, false, false, false, false}, 0},
    {{true, false, false, false, false}, 4},
    {{false, true, false, false, false}, 4},
    {{false, false, true, false, false}, 3},
    {{false, false, false, true, false}, 3},
    {{false, false, false, false, true}, 3},
    {{true, true, false, false, false}, 0},
    {{false, true, true, false, false}, 0},
    {{false, false, true, true, false}, 0},
    {{false, false, false, true, true}, 0},
    {{true, false, false, false, true}, 0},
    {{true, false, true, false, false}, 0},
    {{false, true, false, true, false}, 0},
    {{false, false, true, false, true}, 0},
    {{true, false, false, true, false}, 0},
    {{false, true, false, false, true}, 0},
    {{true, true, false, false, true}, 0},
    {{true, true, true, false, false}, 0},
    {{false, true, true, true, false}, 0},
    {{false, false, true, true, true}, 0},
    {{true, false, false, true, true}, 0},
    {{true, false, true, true, false}, 0},
    {{false, true, false, true, true}, 0},
    {{true, false, true, false, true}, 0},
    {{true, true, false, true, false}, 0},
    {{false, true, true, false, true}, 0},
    {{false, true, true, true, true}, 0},
    {{true, false, true, true, true}, 0},
    {{true, true, false, true, true}, 0},
    {{true, true, true, false, true}, 0},
    {{true, true, true, true, false}, 0},
    {{true, true, true, true, true}, 0},
}};
constexpr std::array<Colors, NUM_COMBINATIONS> COLOR_COMBINATIONS{{
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
Weights generate_weights(Generator& gen) {
    Weights result{};
    std::uniform_real_distribution<float> weight_dist(MIN_WEIGHT, MAX_WEIGHT);
    for (size_t i=0; i < PACKS; i++) {
        for (size_t j=0; j < PACK_SIZE; j++) result[i][j] = weight_dist(gen);
    }
    return result;
}
constexpr size_t WEIGHT_PARAMETER_COUNT = PACK_SIZE * PACKS;
template <size_t Size>
Weights array_to_weights(const std::array<float, Size>& params, size_t start_index) {
    Weights result;
    for (size_t i=0; i < PACKS; i++) {
        for (size_t j=0; j < PACK_SIZE; j++) result[i][j] = params[start_index + i * PACK_SIZE + j];
    }
    return result;
}

template <size_t Size>
void write_weights_to_array(const Weights& weights, std::array<float, Size>& params, size_t start_index) {
    for (size_t i = 0; i < PACKS; i++) {
        for (size_t j=0; j < PACK_SIZE; j++) params[start_index + i * PACK_SIZE + j] = weights[i][j];
    }
}

struct Variables {
    Weights rating_weights = INITIAL_RATING_WEIGHTS;
    Weights colors_weights = INITIAL_COLORS_WEIGHTS;
    Weights fixing_weights = INITIAL_FIXING_WEIGHTS;
    Weights internal_synergy_weights = INITIAL_INTERNAL_SYNERGY_WEIGHTS;
    Weights pick_synergy_weights = INITIAL_PICK_SYNERGY_WEIGHTS;
    Weights openness_weights = INITIAL_OPENNESS_WEIGHTS;
#ifdef OPTIMIZE_RATINGS
    std::array<float, NUM_CARDS> ratings{1};
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

    Variables() = default;
    explicit Variables(std::mt19937_64 & gen) {
        std::uniform_real_distribution<float> rating_dist{0.f, MAX_SCORE};
        std::uniform_real_distribution<float> unit_dist{0.f, 1.f};
        rating_weights = generate_weights(gen);
        colors_weights = generate_weights(gen);
        fixing_weights = generate_weights(gen);
        internal_synergy_weights = generate_weights(gen);
        pick_synergy_weights = generate_weights(gen);
        openness_weights = generate_weights(gen);
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
        rating_weights = array_to_weights(params, 0);
        pick_synergy_weights = array_to_weights(params, WEIGHT_PARAMETER_COUNT);
        fixing_weights = array_to_weights(params, 2 * WEIGHT_PARAMETER_COUNT);
        internal_synergy_weights = array_to_weights(params, 3 * WEIGHT_PARAMETER_COUNT);
        openness_weights = array_to_weights(params, 4 * WEIGHT_PARAMETER_COUNT);
        colors_weights = array_to_weights(params, 5 * WEIGHT_PARAMETER_COUNT);
#ifdef OPTIMIZE_RATINGS
        for (size_t i=0; i < NUM_CARDS; i++) ratings[i] = params[6 * WEIGHT_PARAMETER_COUNT + i];
        constexpr size_t start_index = 6 * WEIGHT_PARAMETER_COUNT + NUM_CARDS;
#else
        constexpr size_t start_index = 6 * WEIGHT_PARAMETER_COUNT;
#endif
        prob_to_include = params[start_index] / 10.1f;
        prob_multiplier = 1 / (1 - prob_to_include);
        similarity_clip = params[start_index + 1] / 10.1f;
        similarity_multiplier = 1 / (1 - similarity_clip);
        is_fetch_multiplier = params[start_index + 2] / 10.f;
        has_basic_types_multiplier = params[start_index + 3] / 10.f;
        is_regular_land_multiplier = params[start_index + 4] / 10.f;
        equal_cards_synergy = params[start_index + 5];
    }

    explicit operator std::array<float, num_parameters>() const {
        std::array<float, num_parameters> result{};
        write_weights_to_array(rating_weights, result, 0);
        write_weights_to_array(pick_synergy_weights, result, WEIGHT_PARAMETER_COUNT);
        write_weights_to_array(fixing_weights, result, 2 * WEIGHT_PARAMETER_COUNT);
        write_weights_to_array(internal_synergy_weights, result, 3 * WEIGHT_PARAMETER_COUNT);
        write_weights_to_array(openness_weights, result, 4 * WEIGHT_PARAMETER_COUNT);
        write_weights_to_array(colors_weights, result, 5 * WEIGHT_PARAMETER_COUNT);
#ifdef OPTIMIZE_RATINGS
        for (size_t i=0; i < NUM_CARDS; i++) result[6 * WEIGHT_PARAMETER_COUNT + i] = ratings[i];
        constexpr size_t start_index = 6 * WEIGHT_PARAMETER_COUNT + NUM_CARDS;
#else
        constexpr size_t start_index = 6 * WEIGHT_PARAMETER_COUNT;
#endif
        result[start_index] = prob_to_include * 10.1f;
        result[start_index + 1] = similarity_clip * 10.1f;
        result[start_index + 2] = is_fetch_multiplier * 10;
        result[start_index + 3] = has_basic_types_multiplier * 10;
        result[start_index + 4] = is_regular_land_multiplier * 10;
        result[start_index + 5] = equal_cards_synergy;
        return result;
    }
};

struct Pick {
    std::array<index_type, MAX_PACK_SIZE> in_pack{std::numeric_limits<index_type>::max()};
    std::array<index_type, MAX_SEEN> seen{std::numeric_limits<index_type>::max()};
    std::array<index_type, MAX_PICKED> picked{std::numeric_limits<index_type>::max()};
    unsigned char pack_num{0};
    unsigned char pick_num{0};
    unsigned char pack_size{0};
    unsigned char packs{0};
    index_type chosen_card{0};
};
extern const std::map<std::string, Colors> FETCH_LANDS;

struct Constants {
    std::array<ColorRequirement, NUM_CARDS> color_requirements{}; // NOLINT(cert-err58-cpp)
    std::array<unsigned char, NUM_CARDS> cmcs{0};
    std::array<Colors, NUM_CARDS> card_colors{{false, false, false, false, false}};
    std::array<bool, NUM_CARDS> is_land{false};
    std::array<bool, NUM_CARDS> is_fetch{false};
    std::array<bool, NUM_CARDS> has_basic_land_types{false};
    std::array<std::array<float, NUM_CARDS>, NUM_CARDS> similarities{{0}};
    CastingProbabilityTable prob_to_cast{{{{{{0}}}}}};
#ifndef OPTIMIZE_RATINGS
    std::array<float, NUM_CARDS> ratings{1};
#endif
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
#endif //DRAFTBOTOPTIMIZATION_DRAFTBOT_OPTIMIZATION_H
