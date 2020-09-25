//
// Created by Devon Richards on 9/4/2020.
//
#include <algorithm>
#include <array>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <vector>

#include "../draftbot_optimization.h"
#include "shared/parameters.h"
#include "shared/util.h"

void average_crossover_weights(const Weights& weights1, const Weights& weights2, std::mt19937_64&, Weights& result) {
    for (size_t pack=0; pack < PACKS; pack++) {
        for (size_t pick=0; pick < PACK_SIZE; pick++) {
            result[pack][pick] = (weights1[pack][pick] + weights2[pack][pick]) / 2;
        }
    }
}

Variables average_crossover_variables(const Variables& variables1, const Variables& variables2, std::mt19937_64& gen) {
    Variables result = variables1;
    average_crossover_weights(variables1.rating_weights, variables2.rating_weights, gen, result.rating_weights);
    average_crossover_weights(variables1.pick_synergy_weights, variables2.pick_synergy_weights, gen, result.pick_synergy_weights);
    average_crossover_weights(variables1.fixing_weights, variables2.fixing_weights, gen, result.fixing_weights);
    average_crossover_weights(variables1.internal_synergy_weights, variables2.fixing_weights, gen, result.internal_synergy_weights);
    average_crossover_weights(variables1.openness_weights, variables2.openness_weights, gen, result.openness_weights);
    average_crossover_weights(variables1.colors_weights, variables2.colors_weights, gen, result.colors_weights);
    result.prob_to_include = (variables1.prob_to_include + variables2.prob_to_include) / 2;
    result.prob_multiplier = 1 / (1 - result.prob_to_include);
    result.similarity_clip = (variables1.similarity_clip + variables2.similarity_clip) / 2;
    result.similarity_multiplier = 1 / (1 - result.similarity_clip);
    result.is_fetch_multiplier = (variables1.is_fetch_multiplier + variables2.is_fetch_multiplier) / 2;
    result.has_basic_types_multiplier = (variables1.has_basic_types_multiplier + variables2.has_basic_types_multiplier) / 2;
    result.is_regular_land_multiplier = (variables1.is_regular_land_multiplier + variables2.is_regular_land_multiplier) / 2;
    result.equal_cards_synergy = (variables1.equal_cards_synergy + variables2.equal_cards_synergy) / 2;
#ifdef OPTIMIZE_RATINGS
    for (size_t i=0; i < NUM_CARDS; i++) result.ratings[i] = (variables1.ratings[i] + variables2.ratings[i]) / 2;
#endif
    return result;
}

template <typename Generator>
void crossover_weights(const Weights& w1, const Weights& w2, Generator& gen, Weights& result) {
    std::uniform_int_distribution<size_t> coin{0, 1};
    for (size_t i=0; i < PACKS; i++) {
        for (size_t j=0; j < PACK_SIZE; j++) {
            if (coin(gen) == 0) result[i][j] = w2[i][j];
            else result[i][j] = w1[i][j];
        }
    }
}

template <typename Generator>
Variables crossover_variables(const Variables& v1, const Variables& v2, Generator& gen) {
    std::uniform_int_distribution<size_t> coin{0, 1};
    Variables result = v1;
    crossover_weights(v1.rating_weights, v2.rating_weights, gen, result.rating_weights);
    crossover_weights(v1.pick_synergy_weights, v2.pick_synergy_weights, gen, result.pick_synergy_weights);
    crossover_weights(v1.fixing_weights, v2.fixing_weights, gen, result.fixing_weights);
    crossover_weights(v1.internal_synergy_weights, v2.internal_synergy_weights, gen, result.internal_synergy_weights);
    crossover_weights(v1.openness_weights, v2.openness_weights, gen, result.openness_weights);
    crossover_weights(v1.colors_weights, v2.colors_weights, gen, result.colors_weights);
#ifdef OPTIMIZE_RATINGS
    for (size_t i=0; i < NUM_CARDS; i++) {
        if (coin(gen) == 0) result.ratings[i] = v2.ratings[i];
    }
#endif
    if (coin(gen) == 0) {
        result.prob_to_include = v2.prob_to_include;
        result.prob_multiplier = v2.prob_multiplier;
    }
    if (coin(gen) == 0) {
        result.similarity_clip = v2.similarity_clip;
        result.similarity_multiplier = v2.similarity_multiplier;
    }
    if (coin(gen) == 0) result.is_fetch_multiplier = v2.is_fetch_multiplier;
    if (coin(gen) == 0) result.has_basic_types_multiplier = v2.has_basic_types_multiplier;
    if (coin(gen) == 0) result.is_regular_land_multiplier = v2.is_regular_land_multiplier;
    if (coin(gen) == 0) result.equal_cards_synergy = v2.equal_cards_synergy;
    return result;
}

Variables optimize_variables(const float temperature, const std::vector<Pick>& picks, const size_t num_generations,
                             const std::shared_ptr<const Constants>& constants,
                             const std::shared_ptr<const Variables>& initial_variables, const size_t seed) {
    std::cout << "Beginning optimize_variables with population size of " << POPULATION_SIZE << " and " << picks.size()
              << " picks, sampling " << PICKS_PER_GENERATION << " per generation" << std::endl << std::endl;
    std::mt19937_64 gen{seed};
    std::uniform_int_distribution<size_t> crossover_selector(0, KEEP_BEST - 1);
    std::uniform_int_distribution<size_t> pick_selector(0, picks.size() - 1);
    std::vector<Variables> population(POPULATION_SIZE);
    if (initial_variables) {
        population[0] = *initial_variables;
        for (size_t i=1; i < POPULATION_SIZE; i++) {
            population[i] = *initial_variables;
            for (size_t j=0; j < NUM_INITIAL_MUTATIONS; j++) mutate_variables(population[i], gen);
        }
    } else {
        for (size_t i=0; i < POPULATION_SIZE; i++) population[i] = Variables(gen);
    }
    std::vector<Variables> old_population(POPULATION_SIZE);
    std::array<std::pair<size_t, double>, POPULATION_SIZE * 2> indexed_losses;
    for (size_t i=0; i < POPULATION_SIZE * 2; i++) indexed_losses[i] = {i, std::numeric_limits<double>::max()};
    std::vector<Pick> chosen_picks(PICKS_PER_GENERATION);
    std::array<std::array<double, 4>, POPULATION_SIZE> old_losses{};
    for (size_t i=0; i < POPULATION_SIZE; i++) old_losses[i] = {std::numeric_limits<double>::max(), std::numeric_limits<double>::max(),
                                                                std::numeric_limits<double>::max(), std::numeric_limits<double>::max()};
    for(size_t generation=0; generation < num_generations; generation++) {
        const auto start = std::chrono::high_resolution_clock::now();
        old_population = population;
        for (size_t i=0; i < POPULATION_SIZE; i++) indexed_losses[POPULATION_SIZE + i] = {POPULATION_SIZE + i, old_losses[i][0]};
        if (generation > 0) {
            for (size_t i = KEEP_BEST; i < POPULATION_SIZE; i++) {
                size_t index1 = crossover_selector(gen);
                size_t index2 = crossover_selector(gen);
                population[i] = crossover_variables(population[index1], population[index2], gen);
            }
            for (Variables &variables : population) {
                mutate_variables(variables, gen);
            }
        }
        for (size_t i=0; i < PICKS_PER_GENERATION; i++) chosen_picks[i] = picks[pick_selector(gen)];
        std::array<std::array<double, 4>, POPULATION_SIZE> losses = run_simulations(population, chosen_picks, temperature, constants);
        std::array<std::pair<std::size_t, double>, POPULATION_SIZE> indexed_accuracies;
        for (size_t i=0; i < POPULATION_SIZE; i++) {
            indexed_losses[i] = {i, losses[i][0]};
            indexed_accuracies[i] = {i, losses[i][2]};
        }
        auto sort_fn = [](const auto& pair1, const auto& pair2){
            if (std::isnan(pair1.second)) return false;
            else if (std::isnan(pair2.second)) return true;
            else return pair1.second < pair2.second;
        };
        std::sort(indexed_losses.begin(), indexed_losses.end(), sort_fn);
        std::sort(indexed_accuracies.begin(), indexed_accuracies.end(), sort_fn);
        auto total_metrics = std::accumulate(losses.begin(), losses.end(), std::array<double, 4>{0, 0, 0, 0},
                                             [](const auto& arr1, const auto& arr2)->std::array<double, 4>{
                                                 return {arr1[0] + arr2[0], arr1[1] + arr2[1], arr1[2] + arr2[2], arr1[3] + arr2[3]};
                                             });
        std::ostringstream out_file_name;
        out_file_name << "output/variables-" << generation << ".json";
        auto output_result = [=](const std::array<double, 4>& arr){ std::cout << "Loss: " << std::setw(WIDTH) << arr[0]
                                                                              << " Categorical Cross-Entropy Loss: " << std::setw(WIDTH) << arr[1]
                                                                              << " Negative Log Accuracy Loss: " << std::setw(WIDTH) << arr[2]
                                                                              << " Accuracy Metric: " << std::setw(WIDTH) << arr[3]
                                                                              << std::endl; };

        std::cout << "Generation " << generation << std::endl;
        if(indexed_losses[0].first >= POPULATION_SIZE) {
            save_variables(old_population[indexed_losses[0].first - POPULATION_SIZE], out_file_name.str());
            std::cout << "Best Loss:      ";
            output_result(old_losses[indexed_losses[0].first - POPULATION_SIZE]);
        } else {
            save_variables(population[indexed_losses[0].first], out_file_name.str());
            std::cout << "Best Loss:      ";
            output_result(losses[indexed_losses[0].first]);
        }
        std::cout << "Best Accuracy:  ";
        output_result(losses[indexed_accuracies[0].first]);
        std::cout << "Worst Survivor: ";
        if (indexed_losses[KEEP_BEST - 1].first >= POPULATION_SIZE) {
            output_result(old_losses[indexed_losses[KEEP_BEST - 1].first - POPULATION_SIZE]);
        } else {
            output_result(losses[indexed_losses[KEEP_BEST - 1].first]);
        }
        std::cout << "Average:        ";
        output_result({total_metrics[0] / POPULATION_SIZE, total_metrics[1] / POPULATION_SIZE,
                       total_metrics[2] / POPULATION_SIZE, total_metrics[3] / POPULATION_SIZE});
        if (generation > 0) {
            std::cout << "Median Loss:    ";
            std::array<double, 4> median = losses[indexed_losses[POPULATION_SIZE].first];
            if (indexed_losses[POPULATION_SIZE].first >= POPULATION_SIZE) {
                median = old_losses[indexed_losses[POPULATION_SIZE].first - POPULATION_SIZE];
            }
            if (indexed_losses[POPULATION_SIZE - 1].first >= POPULATION_SIZE) {
                std::array<double, 4> loss = old_losses[indexed_losses[POPULATION_SIZE - 1].first - POPULATION_SIZE];
                median = {(median[0] + loss[0]) / 2, (median[1] + loss[1]) / 2, (median[2] + loss[2]) / 2,
                          (median[3] + loss[3]) / 2};
            } else {
                std::array<double, 4> loss = losses[indexed_losses[POPULATION_SIZE - 1].first];
                median = {(median[0] + loss[0]) / 2, (median[1] + loss[1]) / 2, (median[2] + loss[2]) / 2,
                          (median[3] + loss[3]) / 2};
            }
            output_result(median);
            std::cout << "Worst Loss:     ";
            if (indexed_losses[2 * POPULATION_SIZE - 1].first >= POPULATION_SIZE) {
                output_result(old_losses[indexed_losses[2 * POPULATION_SIZE - 1].first - POPULATION_SIZE]);
            } else {
                output_result(losses[indexed_losses[2 * POPULATION_SIZE - 1].first]);
            }
        }
        std::vector<Variables> temp_population(POPULATION_SIZE);
        std::array<std::array<double, 4>, POPULATION_SIZE> temp_losses{std::numeric_limits<double>::max()};
        for (size_t i=0; i < POPULATION_SIZE; i++) {
            if (indexed_losses[i].first >= POPULATION_SIZE) {
                temp_population[i] = old_population[indexed_losses[i].first - POPULATION_SIZE];
                temp_losses[i] = old_losses[indexed_losses[i].first - POPULATION_SIZE];
            } else {
                temp_population[i] = population[indexed_losses[i].first];
                temp_losses[i] = losses[indexed_losses[i].first];
            }
        }
        population = temp_population;
        old_losses = temp_losses;
        const auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        std::cout << "Generation took " << diff.count() << " seconds" << std::endl << std::endl;
    }
    return population[0];
}
