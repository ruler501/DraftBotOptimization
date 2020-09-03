//
// Created by Devon Richards on 9/3/2020.
//

#ifndef DRAFTBOTOPTIMIZATION_DRAFTBOT_OPTIMIZATION_H
#define DRAFTBOTOPTIMIZATION_DRAFTBOT_OPTIMIZATION_H
#include <array>
#include <limits>
#include <map>
#include <string>
#include <vector>

constexpr size_t NUM_CARDS = 21467;
constexpr size_t PACKS = 3;
constexpr size_t PACK_SIZE = 15;
constexpr size_t EMBEDDING_SIZE = 64;
constexpr size_t NUM_COLORS = 5;
constexpr size_t POPULATION_SIZE = 1;
constexpr size_t MAX_PACK_SIZE = 32;
constexpr size_t MAX_SEEN = 512;
constexpr size_t MAX_PICKED = 128;
constexpr size_t NUM_COMBINATIONS = 32;
constexpr size_t PICKS_PER_GENERATION = 10'000;
constexpr size_t PROB_DIM_0 = 9;
constexpr size_t PROB_DIM_1 = 7;
constexpr size_t PROB_DIM_2 = 4;
constexpr size_t PROB_DIM_3 = 18;
constexpr size_t PROB_DIM_4 = 18;
constexpr size_t PROB_DIM_5 = 18;
constexpr std::array<char, NUM_COLORS> COLORS{'w', 'u', 'b', 'r', 'g'};
using Weights = std::array<std::array<float, PACK_SIZE>, PACKS>;
using Colors = std::array<bool, NUM_COLORS>;
using Lands = std::array<std::pair<Colors, size_t>, NUM_COMBINATIONS>;
using ColorRequirement = std::pair<std::array<std::pair<std::array<bool, NUM_COMBINATIONS>, size_t>, 5>, size_t>;
using Embedding = std::array<float, EMBEDDING_SIZE>;
using CastingProbabilityTable = std::array<std::array<std::array<std::array<std::array<std::array<float, PROB_DIM_5>, PROB_DIM_4>, PROB_DIM_3>, PROB_DIM_2>, PROB_DIM_1>, PROB_DIM_0>;

constexpr Weights INITIAL_RATING_WEIGHTS{{
                                                 {5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5},
                                                 {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4},
                                                 {3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3},
                                         }};
constexpr Weights INITIAL_COLORS_WEIGHTS{{
                                                 {20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20},
                                                 {40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40},
                                                 {60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60},
                                         }};
constexpr Weights INITIAL_FIXING_WEIGHTS{{
                                                 {0.1f, 0.3f, 0.6f, 0.8f, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                                                 {1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f, 1.5f},
                                                 {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
                                         }};
constexpr Weights INITIAL_INTERNAL_SYNERGY_WEIGHTS{{
                                                           {3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3},
                                                           {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4},
                                                           {5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5},
                                                   }};
constexpr Weights INITIAL_PICK_SYNERGY_WEIGHTS{{
                                                       {3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3},
                                                       {4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4},
                                                       {5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5},
                                               }};
constexpr Weights INITIAL_OPENNESS_WEIGHTS{{
                                                   {4, 12, 12.3f, 12.6f, 13, 13.4f, 13.7f, 14, 15, 14.6f, 14.2f, 13.8f, 13.4f, 13, 12.6f},
                                                   {13, 12.6f, 12.2f, 11.8f, 11.4f, 11, 10.6f, 10.2f, 9.8f, 9.4f, 9, 8.6f, 8.2f, 7.8f, 7},
                                                   {8, 7.5f, 7, 6.5f, 6, 5.5f, 5, 4.5f, 4, 3.5f, 3, 2.5f, 2, 1.5f, 1},
                                           }};
constexpr float INITIAL_PROB_TO_INCLUDE = 0.67f;
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

struct Variables {
    Weights rating_weights = INITIAL_RATING_WEIGHTS;
    Weights colors_weights = INITIAL_COLORS_WEIGHTS;
    Weights fixing_weights = INITIAL_FIXING_WEIGHTS;
    Weights internal_synergy_weights = INITIAL_INTERNAL_SYNERGY_WEIGHTS;
    Weights pick_synergy_weights = INITIAL_PICK_SYNERGY_WEIGHTS;
    Weights openness_weights = INITIAL_OPENNESS_WEIGHTS;
    std::array<float, NUM_CARDS> ratings{1};
    float prob_to_include = INITIAL_PROB_TO_INCLUDE;
    float similarity_clip = INITIAL_SIMILARITY_CLIP;
    float similarity_multiplier = 1 / (1 - INITIAL_SIMILARITY_CLIP);
};

struct Pick {
    std::array<size_t, MAX_PACK_SIZE> in_pack{std::numeric_limits<size_t>::max()};
    std::array<size_t, MAX_SEEN> seen{std::numeric_limits<size_t>::max()};
    std::array<size_t, MAX_PICKED> picked{std::numeric_limits<size_t>::max()};
    size_t pack_num{0};
    size_t pick_num{0};
    size_t pack_size{0};
    size_t packs{0};
    size_t chosen_card{0};
};
extern const std::map<std::string, Colors> FETCH_LANDS;

struct Constants {
    std::array<ColorRequirement, NUM_CARDS> color_requirements{}; // NOLINT(cert-err58-cpp)
    std::array<size_t, NUM_CARDS> cmcs{0};
    std::array<Colors, NUM_CARDS> card_colors{{false, false, false, false, false}};
    std::array<bool, NUM_CARDS> is_land{false};
    std::array<bool, NUM_CARDS> is_fetch{false};
    std::array<bool, NUM_CARDS> has_basic_land_types{false};
    std::array<std::array<float, NUM_CARDS>, NUM_CARDS> similarities{{0}};
    CastingProbabilityTable prob_to_cast{{{{{{0}}}}}};
};

void populate_constants(const std::string& file_name, Constants& constants);
void populate_prob_to_cast(const std::string& file_name, Constants& constants);
std::vector<Pick> load_picks(const std::string& folder);
void save_variables(const Variables& variables, const std::string& file_name);
#endif //DRAFTBOTOPTIMIZATION_DRAFTBOT_OPTIMIZATION_H
