//
// Created by Devon Richards on 9/3/2020.
//
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include <nlohmann/json.hpp>
#include <thread>
#include <future>

#include "draftbot_optimization.h"

const std::map<std::string, std::array<bool, 5>> FETCH_LANDS{ // NOLINT(cert-err58-cpp)
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
std::array<float, NUM_CARDS> INITIAL_RATINGS;

bool does_intersect(const std::array<bool, NUM_COLORS>& a, const std::array<bool, NUM_COLORS>& b) {
    return (a[0] && b[0]) || (a[1] && b[1]) || (a[2] && b[2]) || (a[3] && b[3]) || (a[4] && b[4]);
}

void populate_constants(const std::string& file_name, Constants& constants) {
    std::cout << "populating constants" << std::endl;
    std::ifstream carddb_file(file_name);
    nlohmann::json carddb;
    carddb_file >> carddb;
    std::vector<std::array<float, EMBEDDING_SIZE>> embeddings(NUM_CARDS, {0});
    for (size_t i=0; i < NUM_CARDS; i++) {
        const nlohmann::json card = carddb.at(i);
        constants.cmcs[i] = card["cmc"].get<unsigned char>();
        const auto type_line = card["type"].get<std::string>();
        constants.is_land[i] = type_line.find("Land") != std::string::npos;
        unsigned char basic_land_types = 0;
        for (const std::string& land_type : {"Plains", "Island", "Swamp", "Mountain", "Forest"}) {
            if (type_line.find(land_type) != std::string::npos) basic_land_types++;
        }
        constants.has_basic_land_types[i] = basic_land_types >= BASIC_LAND_TYPES_REQUIRED;
        const auto name = card["name"].get<std::string>();
        const auto fetch_land = FETCH_LANDS.find(name);
        constants.is_fetch[i] = fetch_land != FETCH_LANDS.end();
        if (constants.is_fetch[i]) {
            for (size_t j=0; j < 5; j++) constants.card_colors[i][j] = fetch_land->second[j];
        } else {
            const auto card_color_identity = card["color_identity"].get<std::vector<std::string>>();
            bool our_card_colors[5]{false, false, false, false, false};
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
            for (size_t j=0; j < 5; j++) constants.card_colors[i][j] = our_card_colors[j];
        }
        const auto parsed_cost = card["parsed_cost"].get<std::vector<std::string>>();
        std::map<std::array<bool, NUM_COLORS>, unsigned char> color_requirement_map;
        for (const std::string& symbol : parsed_cost) {
            std::array<bool, NUM_COLORS> colors{false};
            if (symbol.find('p') != std::string::npos || symbol.find('2') != std::string::npos) {
                continue;
            }
            unsigned char count = 0;
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
        ColorRequirement color_requirement{{}, (unsigned char)color_requirement_map.size(), 0};
        unsigned char index = 0;
        for (const auto& [cost_colors, count] : color_requirement_map) {
            if (index >= 5) {
                std::cerr << "Too many color requirements " << color_requirement_map.size() << " for card " << name << std::endl;
            }
            std::array<unsigned char, NUM_COMBINATIONS> valid_lands{0};
            for (size_t j=0; j < NUM_COMBINATIONS; j++) valid_lands[j] = does_intersect(cost_colors, COLOR_COMBINATIONS[j]) ? ~(unsigned char)0 : 0;
            for (size_t j=0; j < NUM_COMBINATIONS; j++) color_requirement.requirements[index].first[j] = valid_lands[j];
            color_requirement.requirements[index].second = count;
            index += 1;
        }
        if (color_requirement_map.size() == 1) {
            color_requirement.offset = ((constants.cmcs[i] << PROB_DIM_1_EXP) | color_requirement.requirements[0].second) << (PROB_DIM_2_EXP + PROB_DIM_3_EXP + PROB_DIM_4_EXP + PROB_DIM_5_EXP);
        } else if (color_requirement_map.size() == 2) {
            size_t required_a = color_requirement.requirements[0].second;
            size_t required_b = color_requirement.requirements[1].second;
            if (required_a < required_b) {
                std::swap(color_requirement.requirements[0], color_requirement.requirements[1]);
                std::swap(required_a, required_b);
            }
            color_requirement.offset = ((((constants.cmcs[i] << PROB_DIM_1_EXP) | required_a) << PROB_DIM_2_EXP) | required_b) << (PROB_DIM_3_EXP + PROB_DIM_4_EXP + PROB_DIM_5_EXP);
        }
        for (size_t j=0; j < color_requirement_map.size(); j++) {
            for (size_t k=0; k < NUM_COMBINATIONS; k++) constants.color_requirements[i].requirements[j].first[k] = color_requirement.requirements[j].first[k];
            constants.color_requirements[i].requirements[j].second = color_requirement.requirements[j].second;
        }
        constants.color_requirements[i].offset = color_requirement.offset;
        const auto elo_iter = card.find("elo");
        if (elo_iter != card.end()) {
            const auto elo = elo_iter->get<float>();
#ifdef OPTIMIZE_RATINGS
            INITIAL_RATINGS[i] = std::min((float) std::pow(10, elo / 400 - 3), MAX_SCORE);
        } else {
            INITIAL_RATINGS[i] = 1.f;
#else
            constants.ratings[i] = std::sqrt(std::pow(10.f, elo / 400 - 4));
//            constants.ratings[i] = std::min((float) std::pow(10, elo / 400 - 4), MAX_SCORE);
        } else {
            constants.ratings[i] = std::sqrt(0.1f);
#endif
        }
        const auto embedding_iter = card.find("embedding");
        if (embedding_iter != card.end() && embedding_iter->size() == 64) {
            embeddings[i] = embedding_iter->get<Embedding>();
        } else {
            embeddings[i] = {0};
        }
    }
    std::array<float, NUM_CARDS> lengths{1};
    for (size_t i=0; i < NUM_CARDS; i++) {
        float length = 0;
        for (size_t j=0; j < EMBEDDING_SIZE; j++) length += embeddings[i][j] * embeddings[i][j];
        lengths[i] = std::sqrt(length);
    }
    std::cout << "Using " << std::thread::hardware_concurrency() << " threads to load data." << std::endl;
    std::vector<std::thread> threads;
    threads.reserve(std::thread::hardware_concurrency());
    for (size_t offset = 0; offset < std::thread::hardware_concurrency(); offset++) {
        threads.emplace_back([offset, &embeddings, &constants, &lengths](){
            for (size_t i=offset; i < NUM_CARDS; i += std::thread::hardware_concurrency()) {
                for (size_t j=0; j < NUM_CARDS; j++) {
                    float dot_product = 0;
                    for (size_t k=0; k < EMBEDDING_SIZE; k++) {
                        dot_product += embeddings[i][k] * embeddings[j][k];
                    }
                    constants.similarities[i][j] = dot_product / lengths[i] / lengths[j];
                    constants.similarities[j][i] = constants.similarities[i][j];
                }
            }
        });
    }
    for (auto& thread : threads) thread.join();
}

void populate_prob_to_cast(const std::string& file_name, Constants& constants) {
    std::cout << "populating prob_to_cast" << std::endl;
    for (float& value : constants.prob_to_cast) value = 0;
    nlohmann::json data;
    std::ifstream data_file(file_name);
    data_file >> data;
    for (const auto& item : data.items()) {
        const size_t cmc = std::stoi(item.key());
        if (PROB_DIM_0 <= cmc) {
            std::cerr << "Too big index at depth 0: " << cmc << std::endl;
            continue;
        }
        for (const auto& item2 : item.value().items()) {
            const size_t required_a = std::stoi(item2.key());
            if (PROB_DIM_1 <= required_a) {
                std::cerr << "Too big index at depth 1: " << required_a << std::endl;
                continue;
            }
            for (const auto& item3 : item2.value().items()) {
                const size_t required_b = std::stoi(item3.key());
                if(PROB_DIM_2 <= required_b) {
                    std::cerr << "Too big index at depth 2: " << required_b << std::endl;
                    continue;
                }
                for (const auto& item4 : item3.value().items()) {
                    const size_t land_count_a = std::stoi(item4.key());
                    if (PROB_DIM_3 <= land_count_a) {
                        std::cerr << "Too big index at depth 3: " << land_count_a << std::endl;
                        continue;
                    }
                    for (const auto& item5 : item4.value().items()) {
                        const size_t land_count_b = std::stoi(item5.key());
                        if (PROB_DIM_4 <= land_count_b) {
                            std::cerr << "Too big index at depth 4: " << land_count_b << std::endl;
                            continue;
                        }
                        for (const auto& item6 : item5.value().items()) {
                            const size_t land_count_ab = std::stoi(item6.key());
                            if (PROB_DIM_5 <= land_count_ab) {
                                std::cerr << "Too big index at depth 5: " << land_count_ab << std::endl;
                                continue;
                            }
                            constants.get_prob_to_cast(cmc, required_a, required_b, land_count_a, land_count_b, land_count_ab) = item6.value().get<float>();
                        }
                    }
                }
            }
        }
    }
}

std::shared_ptr<Pick> parse_pick(const nlohmann::json& pick_json) {
    std::shared_ptr<Pick> result = std::make_shared<Pick>();
    auto _in_pack = pick_json["cardsInPack"];
    if (_in_pack.size() > MAX_PACK_SIZE) {
//        std::cerr << "Pack too big: " << _in_pack.size() << std::endl;
        return {};
    }
    for (size_t i=0; i < _in_pack.size(); i++) {
        auto index = _in_pack.at(i);
        if (!index.is_null()) result->in_pack[i] = index.get<index_type>();
        else return {};
    }
    for (size_t i=_in_pack.size(); i < MAX_PACK_SIZE; i++) result->in_pack[i] = std::numeric_limits<index_type>::max();
    auto _seen = pick_json["seen"];
    if (_seen.size() > MAX_SEEN) {
//        std::cerr << "Seen too big: " << _seen.size() << std::endl;
        return {};
    }
    for (size_t i=0; i < _seen.size(); i++) {
        auto index = _seen.at(i);
        if (!index.is_null()) result->seen[i] = index.get<index_type>();
        else return {};
    }
    for (size_t i=_seen.size(); i < MAX_SEEN; i++) result->seen[i] = std::numeric_limits<index_type>::max();
    auto _picked = pick_json["picked"];
    if (_picked.size() > MAX_PICKED) {
//        std::cerr << "Picked too big: " << _picked.size() << std::endl;
        return {};
    }
    for (size_t i=0; i < _picked.size(); i++) {
        auto index = _picked.at(i);
        if (!index.is_null()) result->picked[i] = index.get<index_type>();
        else return {};
    }
    for (size_t i=_picked.size(); i < MAX_PICKED; i++) result->picked[i] = std::numeric_limits<index_type>::max();
    nlohmann::json pack = pick_json["pack"];
    if (!pack.is_null()) result->pack_num = pack.get<unsigned char>();
    else return {};
    nlohmann::json pick = pick_json["pick"];
    if (!pick.is_null()) result->pick_num = pick.get<unsigned char>();
    else return {};
    nlohmann::json pack_size = pick_json["packSize"];
    if (!pack_size.is_null()) result->pack_size = pack_size.get<unsigned char>();
    else return {};
    nlohmann::json packs = pick_json["packs"];
    if (!packs.is_null()) result->packs = packs.get<unsigned char>();
    else return {};
    nlohmann::json chosen_card = pick_json["chosenCard"];
    if (!chosen_card.is_null()) result->chosen_card = chosen_card.get<index_type>();
    else return {};
    return result;
}

std::vector<Pick> load_picks(const std::string& folder) {
    std::vector<Pick> final_results;
    std::vector<std::future<std::vector<Pick>>> futures;
    std::vector<std::thread> threads;
    threads.reserve(std::thread::hardware_concurrency());
    futures.reserve(std::thread::hardware_concurrency());
    for (size_t offset=0; offset < std::thread::hardware_concurrency(); offset++) {
        std::packaged_task<std::vector<Pick>()> task(
                [offset, &folder]{
            std::vector<Pick> results;
            for(size_t i=offset; i < NUM_PICK_FILES; i += std::thread::hardware_concurrency()) {
                std::string path;
                std::ostringstream path_stream;
                path_stream << folder << i << ".json";
                path = path_stream.str();
                if(i % 53 == 0) std::cout << "Loading picks file " << path << std::endl;
                nlohmann::json drafts;
                {
                    std::ifstream drafts_file(path);
                    drafts_file >> drafts;
                }
                results.reserve(results.size() + drafts.size());
                for (const auto& draft_entry : drafts.items()) {
                    for (const auto& pick_entry : draft_entry.value()["picks"].items()) {
                        if (pick_entry.value()["cardsInPack"].size() > 1) {
                            std::shared_ptr<Pick> pick = parse_pick(pick_entry.value());
                            if (pick) results.push_back(*pick);
                        }
                    }
                }
            }
            return results;
        });
        futures.push_back(task.get_future());
        threads.emplace_back(std::move(task));
    }
    for (size_t i=0; i < threads.size(); i++) {
        threads[i].join();
        futures[i].wait();
        std::vector<Pick> results = futures[i].get();
        final_results.insert(final_results.end(), results.begin(), results.end());
    }
    return final_results;
}

void save_variables(const Variables& variables, const std::string& file_name) {
    nlohmann::json result;
    result["ratingWeights"] = variables.rating_weights;
    result["pickSynergyWeights"] = variables.pick_synergy_weights;
    result["fixingWeights"] = variables.fixing_weights;
    result["internalSynergyWeights"] = variables.internal_synergy_weights;
    result["opennessWeights"] = variables.openness_weights;
    result["colorsWeights"] = variables.colors_weights;
    result["probToInclude"] = variables.prob_to_include;
    result["similarityClip"] = variables.similarity_clip;
#ifdef OPTIMIZE_RATINGS
    result["ratings"] = variables.ratings;
#endif
    result["isFetchMultiplier"] = variables.is_fetch_multiplier;
    result["hasBasicTypesMultiplier"] = variables.has_basic_types_multiplier;
    result["isRegularLandMultiplier"] = variables.is_regular_land_multiplier;
    result["equalCardsSynergy"] = variables.equal_cards_synergy;
    std::ofstream output_file(file_name);
    if (output_file) {
        output_file << result;
        std::cout << "Saved variables to " << file_name << std::endl;
    } else {
        std::cerr << "Failed to open " << file_name << " for writing" << std::endl;
    }
}

Variables load_variables(const std::string& file_name) {
    std::cout << "Loading variables from " << file_name << std::endl;
    Variables result;
    nlohmann::json variables;
    std::ifstream variables_file(file_name);
    variables_file >> variables;
    auto rating_weights = variables["ratingWeights"].get<std::array<std::array<float, PACK_SIZE>, PACKS>>();
    for (size_t i=0; i < PACKS; i++) for (size_t j=0; j < PACK_SIZE; j++) result.rating_weights[i][j] = rating_weights[i][j];
    auto pick_synergy_weights = variables["pickSynergyWeights"].get<std::array<std::array<float, PACK_SIZE>, PACKS>>();
    for (size_t i=0; i < PACKS; i++) for (size_t j=0; j < PACK_SIZE; j++) result.pick_synergy_weights[i][j] = pick_synergy_weights[i][j];
    auto fixing_weights = variables["fixingWeights"].get<std::array<std::array<float, PACK_SIZE>, PACKS>>();
    for (size_t i=0; i < PACKS; i++) for (size_t j=0; j < PACK_SIZE; j++) result.rating_weights[i][j] = rating_weights[i][j];
    auto internal_synergy_weights = variables["internalSynergyWeights"].get<std::array<std::array<float, PACK_SIZE>, PACKS>>();
    for (size_t i=0; i < PACKS; i++) for (size_t j=0; j < PACK_SIZE; j++) result.internal_synergy_weights[i][j] = internal_synergy_weights[i][j];
    auto openness_weights = variables["opennessWeights"].get<std::array<std::array<float, PACK_SIZE>, PACKS>>();
    for (size_t i=0; i < PACKS; i++) for (size_t j=0; j < PACK_SIZE; j++) result.openness_weights[i][j] = openness_weights[i][j];
    auto colors_weights = variables["colorsWeights"].get<std::array<std::array<float, PACK_SIZE>, PACKS>>();
    for (size_t i=0; i < PACKS; i++) for (size_t j=0; j < PACK_SIZE; j++) result.colors_weights[i][j] = colors_weights[i][j];
    result.prob_to_include = variables["probToInclude"].get<float>();
    result.prob_multiplier = 1 / (1 - result.prob_to_include);
    result.similarity_clip = variables["similarityClip"].get<float>();
    result.similarity_multiplier = 1 / (1 - result.similarity_clip);
#ifdef OPTIMIZE_RATINGS
    auto ratings = variables["ratings"].get<std::array<float, NUM_CARDS>>();
    for (size_t i=0; i < NUM_CARDS; i++) result.ratings[i] = ratings[i];
#endif
    result.is_fetch_multiplier = variables["isFetchMultiplier"].get<float>();
    result.has_basic_types_multiplier = variables["hasBasicTypesMultiplier"].get<float>();
    result.is_regular_land_multiplier = variables["isRegularLandMultiplier"].get<float>();
    result.equal_cards_synergy = variables["equalCardsSynergy"].get<float>();
#ifdef OPTIMIZE_RATINGS
    float max_rating = 0;
    for (const float rating : result.ratings) max_rating = std::max(max_rating, rating);
    float scaling_factor = 10.f / max_rating;
    for (size_t i=0; i < NUM_CARDS; i++) result.ratings[i] *= scaling_factor;
    for (size_t i=0; i < PACKS; i++) for (size_t j=0; j < PACK_SIZE; j++) result.rating_weights[i][j] /= scaling_factor;
#endif
    return result;
}
