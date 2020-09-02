#pragma clang diagnostic push
#pragma ide diagnostic ignored "modernize-loop-convert"
//
// Created by Devon Richards on 8/30/2020.
//
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <future>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#ifdef USE_SYCL
#include <CL/sycl.hpp>
#define LOG(x) cl::sycl::log(x)
#define SQRT(x) cl::sycl::sqrt(x)
#define ISNAN(x) cl::sycl::isnan(x)
#define ISINF(x) cl::sycl::isinf(x)
#define EXP(x) cl::sycl::exp(x)
#else
#define LOG(x) std::log(x)
#define SQRT(x) std::sqrt(x)
#define ISNAN(x) std::isnan(x)
#define ISINF(x) std::isinf(x)
#define EXP(x) std::exp(x)
#endif
#include <nlohmann/json.hpp>

constexpr size_t NUM_CARDS = 21467;
constexpr size_t PACKS = 3;
constexpr size_t PACK_SIZE = 15;
constexpr size_t EMBEDDING_SIZE = 64;
constexpr size_t NUM_COLORS = 5;
constexpr size_t POPULATION_SIZE = 32;
constexpr size_t MAX_PACK_SIZE = 32;
constexpr size_t MAX_SEEN = 512;
constexpr size_t MAX_PICKED = 128;
constexpr size_t NUM_COMBINATIONS = 32;
constexpr std::array<char, NUM_COLORS> COLORS{'w', 'u', 'b', 'r', 'g'};
using Weights = std::array<std::array<float, PACK_SIZE>, PACKS>;
using Colors = std::array<bool, NUM_COLORS>;
using Lands = std::array<std::pair<Colors, size_t>, NUM_COMBINATIONS>;
using ColorRequirement = std::array<std::pair<Colors, size_t>, 5>;
using Embedding = std::array<float, EMBEDDING_SIZE>;
using CastingProbabilityTable = std::vector<std::vector<std::vector<std::vector<std::vector<std::vector<float>>>>>>;

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

std::array<float, NUM_CARDS> INITIAL_RATINGS;

struct Variables {
    Weights rating_weights = INITIAL_RATING_WEIGHTS;
    Weights colors_weights = INITIAL_COLORS_WEIGHTS;
    Weights fixing_weights = INITIAL_FIXING_WEIGHTS;
    Weights internal_synergy_weights = INITIAL_INTERNAL_SYNERGY_WEIGHTS;
    Weights pick_synergy_weights = INITIAL_PICK_SYNERGY_WEIGHTS;
    Weights openness_weights = INITIAL_OPENNESS_WEIGHTS;
    std::vector<float> ratings{NUM_CARDS, 1};
    float prob_to_include = INITIAL_PROB_TO_INCLUDE;
    float similarity_clip = INITIAL_SIMILARITY_CLIP;
    float similarity_multiplier = 1 / (1 - INITIAL_SIMILARITY_CLIP);
};

struct Pick {
    std::array<size_t, MAX_PACK_SIZE> in_pack;
    std::array<size_t, MAX_SEEN> seen;
    std::array<size_t, MAX_PICKED> picked;
    size_t pack_num;
    size_t pick_num;
    size_t pack_size;
    size_t packs;
    size_t chosen_card;

    explicit Pick(const nlohmann::json& pick_json) : in_pack{std::numeric_limits<size_t>::max()},
                                                     seen{std::numeric_limits<size_t>::max()},
                                                     picked{std::numeric_limits<size_t>::max()},
                                                     pack_num(pick_json["pack"].get<size_t>()), pick_num(pick_json["pick"].get<size_t>()),
                                                     pack_size(pick_json["packSize"].get<size_t>()), packs(pick_json["packs"].get<size_t>()),
                                                     chosen_card(pick_json["chosenCard"].get<size_t>()) {
        auto _in_pack = pick_json["cardsInPack"].get<std::vector<size_t>>();
        auto _seen = pick_json["seen"].get<std::vector<size_t>>();
        auto _picked = pick_json["picked"].get<std::vector<size_t>>();
        for (size_t i=0; i < _in_pack.size(); i++) in_pack[i] = _in_pack[i];
        for (size_t i=0; i < _seen.size(); i++) seen[i] = _seen[i];
        for (size_t i=0; i < _picked.size(); i++) picked[i] = _picked[i];
    }
};

const std::map<std::string, Colors> FETCH_LANDS{ // NOLINT(cert-err58-cpp)
        {"Arid Mesa", {true, false, false, true, false}},
        {"Bloodstained Mire", {false, false, true, true, false}},
        {"Flooded Strand", {true, true, false, false, false}},
        {"Marsh Flats", {true, false, true, false, false}},
        {"Misty Rainforest", {false, true, false, false, true}},
        {"Polluted Delta", {false, true, true, false, false}},
        {"Scalding Tarn", {false, true, false, true, false}},
        {"Windswept Heath", {true, false, false, false, true}},
        {"Verdant Catacombs", {false, false, true, false, true}},
        {"Wooded Foothills", {false, false, false, true, true}},
        {"Prismatic Vista", {true, true, true, true, true}},
        {"Fabled Passage", {true, true, true, true, true}},
        {"Terramorphic Expanse", {true, true, true, true, true}},
        {"Evolving Wilds", {true, true, true, true, true}},
};

struct Constants {
    std::array<ColorRequirement, NUM_CARDS> color_requirements; // NOLINT(cert-err58-cpp)
    std::array<size_t, NUM_CARDS> cmcs;
    std::array<Colors, NUM_CARDS> card_colors;
    std::array<bool, NUM_CARDS> is_land;
    std::array<bool, NUM_CARDS> is_fetch;
    std::array<bool, NUM_CARDS> has_basic_land_types;
    std::array<Embedding, NUM_CARDS> embeddings;
    CastingProbabilityTable prob_to_cast;
};

void populate_constants(const std::string& file_name, Constants& constants) {
    std::cout << "Parsing " << file_name << std::endl;
    std::ifstream carddb_file(file_name);
    nlohmann::json carddb;
    carddb_file >> carddb;
    std::cout << "populating constants" << std::endl;
    for (size_t i=0; i < NUM_CARDS; i++) {
        const nlohmann::json card = carddb.at(i);
        constants.cmcs[i] = card["cmc"].get<size_t>();
        const auto type_line = card["type"].get<std::string>();
        constants.is_land[i] = type_line.find("Land") != std::string::npos;
        size_t basic_land_types = 0;
        for (const std::string& land_type : {"Plains", "Island", "Swamp", "Mountain", "Forest"}) {
            if (type_line.find(land_type) != std::string::npos) {
                basic_land_types++;
            }
        }
        constants.has_basic_land_types[i] = basic_land_types > 1;
        const auto name = card["name"].get<std::string>();
        const auto fetch_land = FETCH_LANDS.find(name);
        constants.is_fetch[i] = fetch_land != FETCH_LANDS.end();
        if (constants.is_fetch[i]) {
            constants.card_colors[i] = fetch_land->second;
        } else {
            const auto card_color_identity = card["color_identity"].get<std::vector<std::string>>();
            Colors our_card_colors{false, false, false, false, false};
            for (const std::string& color : card_color_identity) {
                if(color == "W") {
                    our_card_colors[0] = true;
                } else if(color == "U") {
                    our_card_colors[1] = true;
                } else if(color == "B") {
                    our_card_colors[2] = true;
                } else if(color == "R") {
                    our_card_colors[3] = true;
                } else if(color == "G") {
                    our_card_colors[4] = true;
                }
            }
            constants.card_colors[i] = our_card_colors;
        }
        const auto parsed_cost = card["parsed_cost"].get<std::vector<std::string>>();
        std::map<Colors, size_t> color_requirement_map;
        for (const std::string& symbol : parsed_cost) {
            Colors colors{false};
            if (symbol.find('p') != std::string::npos || symbol.find('2') != std::string::npos) {
                continue;
            }
            size_t count = 0;
            for (size_t j=0; j < COLORS.size(); j++) {
                if (symbol.find(COLORS[j]) != std::string::npos) {
                    colors[j] = true;
                    count++;
                }
            }
            if(count > 0) {
                auto pair = color_requirement_map.find(colors);
                if (pair != color_requirement_map.end()) {
                    pair->second++;
                } else {
                    color_requirement_map[colors] = 1;
                }
            }
        }
        ColorRequirement color_requirement{{{{false, false, false, false, false}, 0}}};
        size_t index = 0;
        for (const auto& pair : color_requirement_map) {
            color_requirement[index++] = pair;
        }
        constants.color_requirements[i] = color_requirement;
        const auto elo_iter = card.find("elo");
        if (elo_iter != card.end()) {
            const auto elo = elo_iter->get<float>();
            INITIAL_RATINGS[i] = (float) std::pow(10, elo / 400 - 3);
        } else {
            INITIAL_RATINGS[i] = 1.f;
        }
        const auto embedding_iter = card.find("embedding");
        if (embedding_iter != card.end()) {
            constants.embeddings[i] = embedding_iter->get<Embedding>();
        } else {
            constants.embeddings[i] = {0};
        }
    }
    std::cout << "Done populating constants" << std::endl;
}

void populate_prob_to_cast(const std::string& file_name, Constants& constants) {
    std::cout << "Parsing " << file_name << std::endl;
    nlohmann::json data;
    std::ifstream data_file(file_name);
    data_file >> data;
    std::cout << "populating prob_to_cast" << std::endl;
    for (const auto& item : data.items()) {
        const size_t cmc = std::stoi(item.key());
        while (constants.prob_to_cast.size() < cmc + 1) constants.prob_to_cast.emplace_back();
        auto& inner1 = constants.prob_to_cast[cmc];
        for (const auto& item2 : item.value().items()) {
            const size_t required_a = std::stoi(item2.key());
            while(inner1.size() < required_a + 1) inner1.emplace_back();
            auto& inner2 = inner1[required_a];
            for (const auto& item3 : item2.value().items()) {
                const size_t required_b = std::stoi(item3.key());
                while(inner2.size() < required_b + 1) inner2.emplace_back();
                auto& inner3 = inner2[required_b];
                for (const auto& item4 : item3.value().items()) {
                    const size_t land_count_a = std::stoi(item4.key());
                    while (inner3.size() < land_count_a + 1) inner3.emplace_back();
                    auto& inner4 = inner3[land_count_a];
                    for (const auto& item5 : item4.value().items()) {
                        const size_t land_count_b = std::stoi(item5.key());
                        while (inner4.size() < land_count_b + 1) inner4.emplace_back();
                        auto& inner5 = inner4[land_count_b];
                        for (const auto& item6 : item5.value().items()) {
                            const size_t land_count_ab = std::stoi(item6.key());
                            while (inner5.size() < land_count_ab + 1) inner5.emplace_back();
                            inner5[land_count_ab] = item6.value().get<float>();
                        }
                    }
                }
            }
        }
    }
}

const Constants CONSTANTS = [](){ // NOLINT(cert-err58-cpp)
    Constants result;
    populate_constants("data/intToCard.json", result);
    populate_prob_to_cast("data/probTable.json", result);
    return result;
}();

float interpolate_weights(const Weights& weights, const Pick& pick) {
    const float x_index = PACKS * (float)pick.pack_num / (float)pick.packs;
    const float y_index = PACK_SIZE * (float)pick.pick_num / (float)pick.pack_size;
    const auto floor_x_index = (size_t)x_index;
    const auto floor_y_index = (size_t)y_index;
    const auto ceil_x_index = std::min(floor_x_index + 1, PACKS - 1);
    const auto ceil_y_index = std::min(floor_y_index + 1, PACK_SIZE - 1);
    const float x_index_mod_one = x_index - (float)floor_x_index;
    const float y_index_mode_one = y_index - (float)floor_y_index;
    const float inv_x_index_mod_one = 1 - x_index_mod_one;
    const float inv_y_index_mod_one = 1 - y_index_mode_one;
    float XY = x_index_mod_one * y_index_mode_one;
    float Xy = x_index_mod_one * inv_y_index_mod_one;
    float xY = inv_x_index_mod_one * y_index_mode_one;
    float xy = inv_x_index_mod_one * inv_y_index_mod_one;
    float XY_weight = weights[ceil_x_index][ceil_y_index];
    float Xy_weight = weights[ceil_x_index][floor_y_index];
    float xY_weight = weights[floor_x_index][ceil_y_index];
    float xy_weight = weights[floor_x_index][floor_y_index];
    return XY * XY_weight + Xy * Xy_weight + xY * xY_weight + xy * xy_weight;
}

float get_prob_to_cast(size_t cmc, size_t requiredA, size_t requiredB, size_t land_count_a, size_t land_count_b,
                       size_t land_count_ab) {
    if (CONSTANTS.prob_to_cast.size() > cmc) {
        const auto& inner1 = CONSTANTS.prob_to_cast[cmc];
        if (inner1.size() > requiredA) {
            const auto& inner2 = inner1[requiredA];
            if (inner2.size() > requiredB) {
                const auto& inner3 = inner2[requiredB];
                if (inner3.size() > land_count_a) {
                    const auto& inner4 = inner3[land_count_a];
                    if (inner4.size() > land_count_b) {
                        const auto& inner5 = inner4[land_count_b];
                        if (inner5.size() > land_count_ab) {
                            return inner5[land_count_ab];
                        }
                    }
                }
            }
        }
    }
    return 0;
}

bool does_intersect(const Colors& a, const Colors& b) {
    for (size_t i=0; i < NUM_COLORS; i++) {
        if(a[i] && b[i]) return true;
    }
    return false;
}

float get_casting_probability(const Lands& lands, const size_t card_index) {
    const ColorRequirement &color_requirement = CONSTANTS.color_requirements[card_index];
    size_t num_requirements = 5;
    for (size_t i = 0; i < color_requirement.size(); i++) {
        if (color_requirement[i].second == 0) {
            num_requirements = i;
            break;
        }
    }
    if (num_requirements == 0) {
        return 0;
    } else if (num_requirements < 3) {
        const std::pair<Colors, size_t>& first_requirement = color_requirement[0];
        Colors color_a = first_requirement.first;
        size_t required_a = first_requirement.second;
        Colors color_b{false};
        size_t required_b = 0;
        if (num_requirements == 2) {
            const std::pair<Colors, size_t>& second_requirement = color_requirement[1];
            color_b = second_requirement.first;
            required_b = second_requirement.second;
        }
        if (required_a < required_b) {
            const size_t temp = required_a;
            required_a = required_b;
            required_b = temp;
            Colors temp_colors = color_a;
            color_a = color_b;
            color_b = temp_colors;
        }
        size_t land_count_a = 0;
        size_t land_count_b = 0;
        size_t land_count_ab = 0;
        for (size_t i=0; i < NUM_COMBINATIONS; i++) {
            const std::pair<Colors, size_t>& entry = lands[i];
            bool intersection_a = does_intersect(color_a, entry.first);
            bool intersection_b = does_intersect(color_b, entry.first);
            if(!intersection_a && intersection_b) {
                land_count_a++;
            } else if(intersection_a && !intersection_b) {
                land_count_b++;
#pragma clang diagnostic push
#pragma ide diagnostic ignored "ConstantConditionsOC"
            } else if(!intersection_a && !intersection_b) {
                land_count_ab++;
            }
#pragma clang diagnostic pop
        }
        const size_t cmc = CONSTANTS.cmcs[card_index];
        return get_prob_to_cast(cmc, required_a, required_b, land_count_a, land_count_b, land_count_ab);
    } else {
        size_t total_devotion = 0;
        float probability = 1.f;
        const size_t cmc = CONSTANTS.cmcs[card_index];
        for (size_t i=0; i < num_requirements; i++) {
            const std::pair<Colors, size_t>& entry = color_requirement[i];
            total_devotion += entry.second;
            size_t land_count = 0;
            for (size_t j=0; j < NUM_COMBINATIONS; j++) {
                const std::pair<Colors, size_t>& entry2 = lands[j];
                bool intersection = does_intersect(entry.first, entry2.first);
                if (!intersection) {
                    land_count += entry2.second;
                }
            }
            probability *= get_prob_to_cast(cmc, entry.second, 0, land_count, 0, 0);
        }
        size_t land_count = 0;
        for (size_t i=0; i < NUM_COMBINATIONS; i++) {
            const std::pair<Colors, size_t>& entry2 = lands[i];
            for (size_t j=0; j < color_requirement.size(); j++) {
                const std::pair<Colors, size_t>& entry = color_requirement[j];
                bool intersection = does_intersect(entry.first, entry2.first);
                if (!intersection) {
                    land_count += entry2.second;
                    break;
                }
            }
        }
        return probability * get_prob_to_cast(cmc, total_devotion, 0, land_count, 0, 0);
    }
}

float calculate_synergy(const Embedding& embedding1, const Embedding& embedding2, const Variables& variables) {
    float length_embedding1 = 0;
    float length_embedding2 = 0;
    float dot_product = 0;
    for (size_t i=0; i < EMBEDDING_SIZE; i++) {
        length_embedding1 += embedding1[i] * embedding1[i];
        length_embedding2 += embedding2[i] * embedding2[i];
        dot_product += embedding1[i] * embedding2[i];
    }
    const float similarity = dot_product / SQRT(length_embedding1 * length_embedding2);
    const float scaled = variables.similarity_multiplier * std::min(std::max(0.f, similarity - variables.similarity_clip),
                                                                    1 - variables.similarity_clip);
    const float transformed = -LOG(1 - scaled);
    if (ISNAN(transformed)) return 0;
    else if (ISINF(transformed)) return 10;
    else return transformed;
}

float rating_oracle(const size_t card_index, const Lands& lands, const Variables& variables, const Pick& pick) {
    const size_t real_card_index = pick.in_pack[card_index];
    return get_casting_probability(lands, real_card_index) * variables.ratings[real_card_index];
}

float pick_synergy_oracle(const size_t card_index, const Lands& lands, const Variables& variables, const Pick& pick) {
    size_t num_valid_indices = pick.picked.size();
    for (size_t i=0; i < pick.picked.size(); i++) {
        if (pick.picked[i] == std::numeric_limits<size_t>::max()) {
            num_valid_indices = i;
            break;
        }
    }
    if (num_valid_indices == 0) return 0;
    float total_synergy = 0;
    const Embedding& embedding = CONSTANTS.embeddings[pick.in_pack[card_index]];
    for (size_t i=0; i < num_valid_indices; i++) {
        const size_t card = pick.picked[i];
        const float probability = get_casting_probability(lands, card);
        if (probability >= variables.prob_to_include) {
            total_synergy += probability * calculate_synergy(embedding, CONSTANTS.embeddings[card], variables);
        }
    }
    return total_synergy * get_casting_probability(lands, pick.in_pack[card_index]) / pick.picked.size();
}

float fixing_oracle(const size_t card_index, const Lands& lands, const Variables& variables, const Pick& pick) {
    const size_t card_real_index = pick.in_pack[card_index];
    if (CONSTANTS.is_land[card_real_index]) {
        float overlap = 0;
        for (size_t i=0; i < NUM_COLORS; i++){
            if (CONSTANTS.card_colors[card_real_index][i]) {
                size_t count = 0;
                for (size_t j=0; j < NUM_COMBINATIONS; j++) if (lands[j].first[i]) count += lands[j].second;
                if (count > 2) {
                    overlap += 2;
                }
            }
        }
        if (CONSTANTS.is_fetch[card_real_index]) return overlap;
        else if (CONSTANTS.has_basic_land_types[card_real_index]) return 0.75f * overlap;
        else return 0.5f * overlap;
    } else return 0;
}

float internal_synergy_oracle(const size_t, const Lands& lands, const Variables& variables, const Pick& pick) {
    size_t num_valid_indices = pick.picked.size();
    for (size_t i=0; i < pick.picked.size(); i++) {
        if (pick.picked[i] == std::numeric_limits<size_t>::max()) {
            num_valid_indices = i;
            break;
        }
    }
    if (num_valid_indices < 2) return 0;
    float internal_synergy = 0;
    std::array<float, MAX_PICKED> probabilities{0};
    for (size_t i=0; i < num_valid_indices; i++) probabilities[i] = get_casting_probability(lands, pick.picked[i]);
    for(size_t i=0; i < num_valid_indices; i++) {
        const float probability = probabilities[i];
        if (probability >= variables.prob_to_include) {
            const Embedding& embedding = CONSTANTS.embeddings[pick.picked[i]];
            float card_synergy = 0;
            for (size_t j = 0; j < i; j++) {
                const float probability2 = probabilities[j];
                if (probability2 >= variables.prob_to_include) {
                    card_synergy += probability2 * calculate_synergy(embedding, CONSTANTS.embeddings[pick.picked[j]], variables);
                }
            }
            internal_synergy += probability * card_synergy;
        }
    }
    return internal_synergy / (float)(pick.picked.size() * (pick.picked.size() + 1));
}

template<size_t Size>
float sum_gated_rating(const Lands& lands, const Variables& variables, const std::array<size_t, Size>& indices) {
    float result = 0;
    size_t num_valid_indices = indices.size();
    for (size_t i=0; i < indices.size(); i++) {
        if (indices[i] == std::numeric_limits<size_t>::max()) {
            num_valid_indices = i;
            break;
        }
    }
    for (size_t i=0; i < num_valid_indices; i++) {
        const size_t index = indices[i];
        const float probability = get_casting_probability(lands, index);
        if (probability >= variables.prob_to_include) {
            result += variables.ratings[index] * probability;
        }
    }
    return result;
}

float openness_oracle(const size_t, const Lands& lands, const Variables& variables, const Pick& pick) {
    return sum_gated_rating(lands, variables, pick.seen);
}

float colors_oracle(const size_t, const Lands& lands, const Variables& variables, const Pick& pick) {
    return sum_gated_rating(lands, variables, pick.picked);
}

float get_score(const size_t card_index, const Lands& lands, const Variables& variables, const Pick& pick,
                const float rating_weight, const float pick_synergy_weight, const float fixing_weight,
                const float internal_synergy_weight, const float openness_weight, const float colors_weight) {
    const float rating_score = rating_oracle(card_index, lands, variables, pick);
//     std::cout << rating_score << "*" << rating_weight;
    const float pick_synergy_score = pick_synergy_oracle(card_index, lands, variables, pick);
//     std::cout << " + " << pick_synergy_score << "*" << pick_synergy_weight;
    const float fixing_score = fixing_oracle(card_index, lands, variables, pick);
//     std::cout << " + " << fixing_score << "*" << fixing_weight;
    const float internal_synergy_score = internal_synergy_oracle(card_index, lands, variables, pick);
//     std::cout << " + " << internal_synergy_score << "*" << internal_synergy_weight;
    const float openness_score = openness_oracle(card_index, lands, variables, pick);
//     std::cout << " + " << openness_score << "*" << openness_weight;
    const float colors_score = colors_oracle(card_index, lands, variables, pick);
//     std::cout << " + " << colors_score << "*" << colors_weight << std::endl;
    return rating_score*rating_weight + pick_synergy_score*pick_synergy_weight + fixing_score*fixing_weight
           + internal_synergy_score*internal_synergy_weight + openness_score*openness_weight + colors_score*colors_weight;
}

float do_climb(const size_t card_index, const Variables& variables, const Pick& pick) {
    float previous_score = -1;
    float current_score = 0;
    const float rating_weight = interpolate_weights(variables.rating_weights, pick);
    const float pick_synergy_weight = interpolate_weights(variables.pick_synergy_weights, pick);
    const float fixing_weight = interpolate_weights(variables.fixing_weights, pick);
    const float internal_synergy_weight = interpolate_weights(variables.internal_synergy_weights, pick);
    const float openness_weight = interpolate_weights(variables.openness_weights, pick);
    const float colors_weight = interpolate_weights(variables.colors_weights, pick);
    Lands lands = DEFAULT_LANDS;
    while (previous_score < current_score) {
        previous_score = current_score;
        for(size_t remove_index=1; remove_index < COLORS.size() + 1; remove_index++) {
            if (lands[remove_index].second > 0) {
                bool breakout = false;
                for (size_t add_index=1; add_index < COLORS.size() + 1; add_index++) {
                    Lands new_lands = lands;
                    new_lands[remove_index].second -= 1;
                    new_lands[add_index].second += 1;
                    float score = get_score(card_index, new_lands, variables, pick, rating_weight, pick_synergy_weight,
                                            fixing_weight, internal_synergy_weight, openness_weight, colors_weight);
                    if (score > current_score) {
                        current_score = score;
                        lands = new_lands;
                        breakout = true;
                        break;
                    }
                }
                if (breakout) break;
            }
        }
    }
    return current_score;
}

float calculate_loss(const Pick& pick, const Variables& variables, const float temperature) {
    std::array<double, MAX_PACK_SIZE> scores{0};
    std::array<float, MAX_PACK_SIZE> softmaxed{0};
    double denominator = 0;
    size_t num_valid_indices = pick.in_pack.size();
    for (size_t i=0; i < pick.in_pack.size(); i++) {
        if (pick.in_pack[i] == std::numeric_limits<size_t>::max()) {
            num_valid_indices = i;
            break;
        }
    }
    for (size_t i=0; i < num_valid_indices; i++) {
        try {
            scores[i] = EXP((double) do_climb(i, variables, pick) / temperature);
        } catch (std::exception& exc) {
            std::cerr << exc.what() << std::endl;
        }
        denominator += scores[i];
    }
    for (size_t i=0; i < num_valid_indices; i++) softmaxed[i] = (float)(scores[i] / denominator);
    size_t match_count = 0;
    std::array<bool, MAX_PACK_SIZE> match_picked{false};
    for (size_t i=0; i < num_valid_indices; i++) {
        if (pick.in_pack[i] == pick.chosen_card) {
            match_picked[i] = true;
            match_count++;
        }
    }
    const float matched_weight = 1.f / match_count;
    float loss = 0;
    for(size_t i=0; i < num_valid_indices; i++) {
        if (match_picked[i]) {
            if (scores[i] > 0) {
                loss += matched_weight * LOG(softmaxed[i]);
            } else {
                return 0;
            }
        }
    }
    return -loss;
}

template<typename PickPtr>
float get_batch_loss(const PickPtr picks, const Variables& variables, const float temperature, const size_t picks_size) {
    float sum_loss = 0;
    for (size_t i=0; i < picks_size; i++) {
        const float pick_loss = calculate_loss(picks[i], variables, temperature);
        if (pick_loss < 0) return -1;
        sum_loss += pick_loss;
    }
    return sum_loss / picks_size;
}

class calculate_batch_loss;

std::array<float, POPULATION_SIZE> run_simulations(const std::vector<Variables>& variables,
                                                   const std::vector<Pick>& picks, const float temperature) {
    std::array<float, POPULATION_SIZE> results{0};
#ifdef USE_SYCL
    using namespace cl::sycl;
    try {
        const std::size_t plat_index = 1;
        const std::size_t dev_index = std::numeric_limits<std::size_t>::max();
        const auto dev_type = cl::sycl::info::device_type::gpu;
        std::cout << "Getting platforms" << std::endl;
        // Platform selection
        auto plats = cl::sycl::platform::get_platforms();
        std::cout << "Retrieved plats" << std::endl;
        std::cout << "Empty: " << plats.empty() << std::endl;
        if (plats.empty()) throw std::runtime_error{ "No OpenCL platform found." };

        std::cout << "Found platforms:" << std::endl;
        for (const auto& plat : plats) std::cout << "\t" << plat.get_info<cl::sycl::info::platform::vendor>() << std::endl;

        auto plat = plats.at(plat_index == std::numeric_limits<std::size_t>::max() ? 0 : plat_index);

        std::cout << "\n" << "Selected platform: " << plat.get_info<cl::sycl::info::platform::vendor>() << std::endl;

        // Device selection
        auto devs = plat.get_devices(dev_type);
        std::cout << "Found devices:"<< std::endl;
        for (const auto& dev : devs) std::cout << "\t" << dev.get_info<cl::sycl::info::device::name>() << std::endl;

        if (devs.empty()) throw std::runtime_error{ "No OpenCL device of specified type found on selected platform." };

        auto dev = devs.at(dev_index == std::numeric_limits<std::size_t>::max() ? 0 : dev_index);

        std::cout << "Selected device: " << dev.get_info<cl::sycl::info::device::name>() << "\n" << std::endl;

        // Context, queue, buffer creation
        auto async_error_handler = [](const cl::sycl::exception_list& errors) { for (auto error : errors) throw error; };

        cl::sycl::context ctx{ dev, async_error_handler };

        cl::sycl::gpu_selector selector;

        cl::sycl::queue myQueue{selector};

        /* Create a scope to control data synchronisation of buffer objects. */
        {
            buffer<Variables, 1> inputBuffer(variables.data(), sycl::range<1>(POPULATION_SIZE));
            buffer<Pick, 1> picksBuffer(picks.data(), sycl::range<1>(picks.size()));
            buffer<float, 1> outputBuffer(results.data(), sycl::range<1>(POPULATION_SIZE));
            size_t picksSize = picks.size();

            /* Submit a command_group to execute from the queue. */
            myQueue.submit([&](handler &cgh) {

                /* Create accessors for accessing the input and output data within the
                 * kernel. */
                auto inputPtr = inputBuffer.get_access<access::mode::read>(cgh);
                auto pickPtr = picksBuffer.get_access<access::mode::read>(cgh);
                auto outputPtr = outputBuffer.get_access<access::mode::write>(cgh);

                cgh.parallel_for<class calculate_batch_loss>(sycl::range<1>(POPULATION_SIZE),
                                                             [=](cl::sycl::id<1> wiID) {
                                                                 outputPtr[wiID] = get_batch_loss(pickPtr, inputPtr[wiID], temperature, picksSize);
                                                             }
                );
            });
            myQueue.wait();
        }

    } catch (exception& e) {

        /* In the case of an exception being throw, print the error message and
         * rethrow it. */
        std::cout << e.what();
        throw e;
    }
#else
    std::array<std::future<float>, POPULATION_SIZE> futures;
    std::array<std::thread, POPULATION_SIZE> threads;
    for (size_t i=0; i < POPULATION_SIZE; i++) {
        std::packaged_task<float()> task(
                [&variables, i, &picks, temperature] { return get_batch_loss(picks, variables[i], temperature, picks.size()); }); // wrap the function
        futures[i] = std::move(task.get_future());  // get a future
        std::thread t(std::move(task)); // launch on a thread
        threads[i] = std::move(t);
    }
    for (size_t i=0; i < POPULATION_SIZE; i++) {
        threads[i].join();
        futures[i].wait();
        results[i] = futures[i].get();
    }
//    for (size_t i=0; i < POPULATION_SIZE; i++) {
//        results[i] = get_batch_loss(picks, variables[i], temperature, picks.size());
//    }
#endif
    return results;
}

std::vector<Pick> load_picks(const std::string& folder) {
    std::cout << "Loading picks" << std::endl;
    std::vector<Pick> results;
    for (size_t i=1; i < 6; i++) {
        std::ostringstream path_stream;
        path_stream << folder << i << ".json";
        std::string path = path_stream.str();
        std::cout << "Loading file " << path << std::endl;
        nlohmann::json drafts;
        std::ifstream drafts_file(path);
        drafts_file >> drafts;
        std::cout << "Parsed file" << std::endl;
        for (const auto& draft_entry : drafts.items()) {
            for (const auto& pick_entry : draft_entry.value()["picks"].items()) {
                if (pick_entry.value()["cardsInPack"].size() > 1) {
                    results.emplace_back(pick_entry.value());
                }
            }
        }
    }
    std::cout << "Done loading picks" << std::endl;
    return results;
}

Weights crossover_weights(const Weights& weights1, const Weights& weights2, std::mt19937_64& gen) {
    std::uniform_int_distribution<size_t> coin(0, 1);
    Weights result = weights1;
    for (size_t pack=0; pack < PACKS; pack++) {
        for (size_t pick=0; pick < PACK_SIZE; pick++) {
            if (coin(gen) == 1) result[pack][pick] = weights2[pack][pick];
        }
    }
    return result;
}

Variables crossover_variables(const Variables& variables1, const Variables& variables2, std::mt19937_64& gen) {
    std::uniform_int_distribution<size_t> coin(0, 1);
    Variables result = variables1;
    result.rating_weights = crossover_weights(variables1.rating_weights, variables2.rating_weights, gen);
    result.pick_synergy_weights = crossover_weights(variables1.pick_synergy_weights, variables2.pick_synergy_weights, gen);
    result.fixing_weights = crossover_weights(variables1.fixing_weights, variables2.fixing_weights, gen);
    result.internal_synergy_weights = crossover_weights(variables1.internal_synergy_weights, variables2.fixing_weights, gen);
    result.openness_weights = crossover_weights(variables1.openness_weights, variables2.openness_weights, gen);
    result.colors_weights = crossover_weights(variables1.colors_weights, variables2.colors_weights, gen);
    if (coin(gen) == 1) result.prob_to_include = variables2.prob_to_include;
    if (coin(gen) == 1) {
        result.similarity_clip = variables2.similarity_clip;
        result.similarity_multiplier = variables2.similarity_multiplier;
    }
    for (size_t i=0; i < NUM_CARDS; i++) if (coin(gen)) result.ratings[i] = variables2.ratings[i];
    return result;
}

Weights mutate_weights(Weights& weights, std::mt19937_64& gen) {
    std::normal_distribution<float> std_dev_2{0, 2.0};
    std::uniform_int_distribution<size_t> int_distribution(0, 14);
    for (auto& pack : weights) {
        for (float& weight : pack) {
            if (int_distribution(gen) == 0) weight += std_dev_2(gen);
        }
    }
    return weights;
}

Variables mutate_variables(Variables& variables, std::mt19937_64& gen) {
    std::normal_distribution<float> std_dev_005{0, 0.05f};
    std::normal_distribution<float> std_dev_05{0, 0.5f};
    std::uniform_int_distribution<size_t> int_distribution(0, 9);
    std::uniform_int_distribution<size_t> int_distribution2(0, 44);
    mutate_weights(variables.rating_weights, gen);
    mutate_weights(variables.pick_synergy_weights, gen);
    mutate_weights(variables.fixing_weights, gen);
    mutate_weights(variables.internal_synergy_weights, gen);
    mutate_weights(variables.openness_weights, gen);
    mutate_weights(variables.colors_weights, gen);
    if(int_distribution(gen) == 0) {
        variables.prob_to_include += std_dev_005(gen);
        variables.prob_to_include = std::max(std::min(variables.prob_to_include, 1.f), 0.f);
    }
    if(int_distribution(gen) == 0) {
        variables.similarity_clip += std_dev_005(gen);
        variables.similarity_clip = std::max(std::min(variables.similarity_clip, 0.99f), 0.f);
        variables.similarity_multiplier = 1 / (1 - variables.similarity_clip);
    }
    for (float& rating : variables.ratings) {
        if(int_distribution2(gen) == 0) rating += std_dev_05(gen);
    }
    return variables;
}

Variables optimize_variables(const float temperature, const std::vector<Pick>& picks, const size_t num_generations) {
    std::cout << "Beginning optimize_variables" << std::endl;
    std::random_device rd{};
    std::mt19937_64 gen{rd()};
    const std::uniform_int_distribution<size_t> crossover_selector(0, POPULATION_SIZE / 2 + POPULATION_SIZE % 2 - 1);
    std::vector<Variables> population(POPULATION_SIZE);
    std::cout << "Initializing population_indices" << std::endl;
    std::array<size_t, POPULATION_SIZE> population_indices{0};
    std::vector<float> initial_ratings_vector(INITIAL_RATINGS.begin(), INITIAL_RATINGS.end());
    std::cout << "Ratings size: " << initial_ratings_vector.size();
    for (size_t i=0; i < POPULATION_SIZE; i++) {
        population_indices[i] = i;
        population[i].ratings = initial_ratings_vector;
    }
    for(int generation=0; generation < num_generations; generation++) {
        std::cout << "Beginning crossover" << std::endl;
        for (size_t i=0; i < POPULATION_SIZE / 2 + POPULATION_SIZE % 2; i++) {
            size_t index1 = crossover_selector(gen);
            size_t index2 = crossover_selector(gen);
            population[POPULATION_SIZE / 2 + i] = crossover_variables(population[index1], population[index2], gen);
        }
        std::cout << "Beginning mutation" << std::endl;
        for (Variables& variables : population) {
            mutate_variables(variables, gen);
        }
        std::cout << "Beginning calculating losses" << std::endl;
        const auto start = std::chrono::high_resolution_clock::now();
        std::array<float, POPULATION_SIZE> losses = run_simulations(population, picks, temperature);
        const auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        std::cout << "Generation took " << diff.count() << " seconds" << std::endl;
        std::array<std::pair<std::size_t, float>, POPULATION_SIZE> indexed_losses;
        for (size_t i=0; i < POPULATION_SIZE; i++) indexed_losses[i] = {i, losses[i]};
        std::sort(indexed_losses.begin(), indexed_losses.end(),
                  [](const auto& pair1, const auto& pair2){ return pair1.second < pair2.second; });
        std::cout << "Generation: " << generation << " Best Loss: " << indexed_losses[0].second << std::endl;
        std::vector<Variables> temp_population(POPULATION_SIZE);
        for (size_t i=0; i < POPULATION_SIZE; i++) temp_population[i] = population[indexed_losses[i].first];
        population = temp_population;
    }
    return population[0];
}

int main(const int argc, const char* argv[]) {
    if (argc != 3) {
        return -1;
    }
    const float temperature = std::strtof(argv[1], nullptr);
    const size_t generations = std::strtoull(argv[2], nullptr, 10);
    const std::vector<Pick> all_picks = load_picks("data/drafts/");
    std::cout << "created all_picks" << std::endl;
    const Variables best_variables = optimize_variables(temperature, all_picks, generations);
    nlohmann::json result;
    result["ratingWeights"] = best_variables.rating_weights;
    result["pickSynergyWeights"] = best_variables.pick_synergy_weights;
    result["fixingWeights"] = best_variables.fixing_weights;
    result["internalSynergyWeights"] = best_variables.internal_synergy_weights;
    result["opennessWeights"] = best_variables.openness_weights;
    result["colorsWeights"] = best_variables.colors_weights;
    result["probToInclude"] = best_variables.prob_to_include;
    result["similarityClip"] = best_variables.similarity_clip;
    result["ratings"] = best_variables.ratings;
    std::ofstream output_file("output/variables.json");
    output_file << result;
}
#pragma clang diagnostic pop

