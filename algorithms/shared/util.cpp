//
// Created by Devon Richards on 9/6/2020.
//
#include <random>

#include "../../draftbot_optimization.h"

Weights mutate_weights(Weights& weights, std::mt19937_64& gen) {
    std::normal_distribution<float> std_dev_weights{0, WEIGHT_VOLATILITY};
    std::uniform_int_distribution<size_t> int_distribution(0, WEIGHT_INV_PROB_TO_CHANGE - 1);
    for (auto& pack : weights) {
        for (float& weight : pack) {
            if (int_distribution(gen) == 0) {
                weight += std_dev_weights(gen);
                weight = std::max(std::min(weight, MAX_WEIGHT), MIN_WEIGHT);
            }
        }
    }
    return weights;
}

Variables mutate_variables(Variables& variables, std::mt19937_64& gen) {
    std::normal_distribution<float> std_dev_clip{0, CLIP_VOLATILITY};
    std::normal_distribution<float> std_dev_rating{0, RATING_VOLATILITY};
    std::normal_distribution<float> std_dev_multiplier{0, MULTIPLIER_VOLATILITY};
    std::uniform_int_distribution<size_t> int_distribution_clip(0, CLIP_INV_PROB_TO_CHANGE - 1);
    std::uniform_int_distribution<size_t> int_distribution_rating(0, RATING_INV_PROB_TO_CHANGE - 1);
    std::uniform_int_distribution<size_t> int_distribution_multiplier(0, MULTIPLIER_INV_PROB_TO_CHANGE - 1);
    std::uniform_int_distribution<size_t> int_distribution_equal_cards(0, EQUAL_CARDS_INV_PROB_TO_CHANGE - 1);
    mutate_weights(variables.rating_weights, gen);
    mutate_weights(variables.pick_synergy_weights, gen);
    mutate_weights(variables.fixing_weights, gen);
    mutate_weights(variables.internal_synergy_weights, gen);
    mutate_weights(variables.openness_weights, gen);
    mutate_weights(variables.colors_weights, gen);
    if (int_distribution_clip(gen) == 0) {
        variables.prob_to_include += std_dev_clip(gen);
        variables.prob_to_include = std::max(std::min(variables.prob_to_include, 0.99f), 0.f);
        variables.prob_multiplier = 1 / (1 - variables.prob_to_include);
    }
    if (int_distribution_clip(gen) == 0) {
        variables.similarity_clip += std_dev_clip(gen);
        variables.similarity_clip = std::max(std::min(variables.similarity_clip, 0.99f), 0.f);
        variables.similarity_multiplier = 1 / (1 - variables.similarity_clip);
    }
    if (int_distribution_multiplier(gen) == 0) {
        variables.is_fetch_multiplier += std_dev_multiplier(gen);
        variables.is_fetch_multiplier = std::max(std::min(variables.is_fetch_multiplier, 1.f), 0.f);
    }
    if (int_distribution_multiplier(gen) == 0) {
        variables.has_basic_types_multiplier += std_dev_multiplier(gen);
        variables.has_basic_types_multiplier = std::max(std::min(variables.has_basic_types_multiplier, 1.f), 0.f);
    }
    if (int_distribution_multiplier(gen) == 0) {
        variables.is_regular_land_multiplier += std_dev_multiplier(gen);
        variables.is_regular_land_multiplier = std::max(std::min(variables.is_regular_land_multiplier, 1.f), 0.f);
    }
    if (int_distribution_equal_cards(gen) == 0) {
        variables.equal_cards_synergy += std_dev_rating(gen);
        variables.equal_cards_synergy = std::max(std::min(variables.equal_cards_synergy, MAX_SCORE), 0.f);
    }
    for (size_t i=0; i < NUM_CARDS; i++) {
        if(int_distribution_rating(gen) == 0) {
            variables.ratings[i] += std_dev_rating(gen);
            variables.ratings[i] = std::max(std::min(MAX_SCORE, variables.ratings[i]), 0.f);
        }
    }
    return variables;
}