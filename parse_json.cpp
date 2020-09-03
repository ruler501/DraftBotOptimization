//
// Created by Devon Richards on 9/3/2020.
//
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include <nlohmann/json.hpp>

#include "draftbot_optimization.h"

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
std::array<float, NUM_CARDS> INITIAL_RATINGS;

bool does_intersect(const Colors& a, const Colors& b) {
    return (a[0] && b[0]) || (a[1] && b[1]) || (a[2] && b[2]) || (a[3] && b[3]) || (a[4] && b[4]);
}

void populate_constants(const std::string& file_name, Constants& constants) {
    std::cout << "populating constants" << std::endl;
    std::ifstream carddb_file(file_name);
    nlohmann::json carddb;
    carddb_file >> carddb;
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
        ColorRequirement color_requirement{{{{false}, 0}}};
        size_t index = 0;
        for (const auto& pair : color_requirement_map) {
            std::array<bool, 32> valid_lands{false};
            for (size_t j=0; j < 32; j++) valid_lands[j] = does_intersect(pair.first, COLOR_COMBINATIONS[j]);
            color_requirement[index] = {valid_lands, pair.second};
            index += 1;
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
    float rating_max = 0.f;
    for (size_t i=0; i < NUM_CARDS; i++) rating_max = std::max(rating_max, INITIAL_RATINGS[i]);
    rating_max = 10 / rating_max;
    for (size_t i=0; i < NUM_CARDS; i++) rating_max /= INITIAL_RATINGS[i];
}

void populate_prob_to_cast(const std::string& file_name, Constants& constants) {
    std::cout << "populating prob_to_cast" << std::endl;
    nlohmann::json data;
    std::ifstream data_file(file_name);
    data_file >> data;
    for (const auto& item : data.items()) {
        const size_t cmc = std::stoi(item.key());
        if (constants.prob_to_cast.size() < cmc + 1) {
            std::cerr << "Too big index at depth 0: " << cmc << std::endl;
            continue;
        }
        auto& inner1 = constants.prob_to_cast[cmc];
        for (const auto& item2 : item.value().items()) {
            const size_t required_a = std::stoi(item2.key());
            if (inner1.size() < required_a + 1) {
                std::cerr << "Too big index at depth 1: " << required_a << std::endl;
                continue;
            }
            auto& inner2 = inner1[required_a];
            for (const auto& item3 : item2.value().items()) {
                const size_t required_b = std::stoi(item3.key());
                if(inner2.size() < required_b + 1) {
                    std::cerr << "Too big index at depth 2: " << required_b << std::endl;
                    continue;
                }
                auto& inner3 = inner2[required_b];
                for (const auto& item4 : item3.value().items()) {
                    const size_t land_count_a = std::stoi(item4.key());
                    if (inner3.size() < land_count_a + 1) {
                        std::cerr << "Too big index at depth 3: " << land_count_a << std::endl;
                        continue;
                    }
                    auto& inner4 = inner3[land_count_a];
                    for (const auto& item5 : item4.value().items()) {
                        const size_t land_count_b = std::stoi(item5.key());
                        if (inner4.size() < land_count_b + 1) {
                            std::cerr << "Too big index at depth 4: " << land_count_b << std::endl;
                            continue;
                        }
                        auto& inner5 = inner4[land_count_b];
                        for (const auto& item6 : item5.value().items()) {
                            const size_t land_count_ab = std::stoi(item6.key());
                            if (inner5.size() < land_count_ab + 1) {
                                std::cerr << "Too big index at depth 5: " << land_count_ab << std::endl;
                                continue;
                            }
                            inner5[land_count_ab] = item6.value().get<float>();
                        }
                    }
                }
            }
        }
    }
}

Pick parse_pick(nlohmann::json pick_json) {
    auto _in_pack = pick_json["cardsInPack"].get<std::vector<size_t>>();
    auto _seen = pick_json["seen"].get<std::vector<size_t>>();
    auto _picked = pick_json["picked"].get<std::vector<size_t>>();
    std::array<size_t, MAX_PACK_SIZE> in_pack{std::numeric_limits<size_t>::max()};
    std::array<size_t, MAX_SEEN> seen{std::numeric_limits<size_t>::max()};
    std::array<size_t, MAX_PICKED> picked{std::numeric_limits<size_t>::max()};
    for (size_t i=0; i < _in_pack.size(); i++) in_pack[i] = _in_pack[i];
    for (size_t i=_in_pack.size(); i < in_pack.size(); i++) in_pack[i] = std::numeric_limits<size_t>::max();
    for (size_t i=0; i < _seen.size(); i++) seen[i] = _seen[i];
    for (size_t i=_seen.size(); i < seen.size(); i++) seen[i] = std::numeric_limits<size_t>::max();
    for (size_t i=0; i < _picked.size(); i++) picked[i] = _picked[i];
    for (size_t i=_picked.size(); i < picked.size(); i++) picked[i] = std::numeric_limits<size_t>::max();
    return {in_pack, seen, picked, pick_json["pack"].get<size_t>(), pick_json["pick"].get<size_t>(),
            pick_json["packSize"].get<size_t>(), pick_json["packs"].get<size_t>(), pick_json["chosenCard"].get<size_t>()};
}

std::vector<Pick> load_picks(const std::string& folder) {
    std::vector<Pick> results;
    for (size_t i=1; i < 6; i++) {
        std::ostringstream path_stream;
        path_stream << folder << i << ".json";
        std::string path = path_stream.str();
        std::cout << "Loading picks file " << path << std::endl;
        nlohmann::json drafts;
        std::ifstream drafts_file(path);
        drafts_file >> drafts;
        for (const auto& draft_entry : drafts.items()) {
            for (const auto& pick_entry : draft_entry.value()["picks"].items()) {
                if (pick_entry.value()["cardsInPack"].size() > 1) {
                    results.push_back(parse_pick(pick_entry.value()));
                }
            }
        }
    }
    return results;
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
    result["ratings"] = variables.ratings;
    std::ofstream output_file(file_name);
    output_file << result;
}