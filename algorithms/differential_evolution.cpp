//
// Created by Devon Richards on 9/6/2020.
//
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
#include "shared/util.h"

template <typename Generator>
std::array<size_t, 3> pick_distinct(size_t base, Generator& gen) {
    std::array<size_t, 3> result{};
    std::uniform_int_distribution<size_t> first_dist{0, POPULATION_SIZE - 2};
    result[0] = first_dist(gen);
    size_t min = result[0];
    size_t max = base;
    if (result[0] >= base) {
        result[0] += 1;
        min = base;
        max = result[0];
    }
    std::uniform_int_distribution<size_t> second_dist{0, POPULATION_SIZE - 3};
    result[1] = second_dist(gen);
    size_t middle;
    if (result[1] >= min) {
        if (result[1] + 1 >= max) {
            result[1] += 2;
            middle = max;
            max = result[1];
        } else {
            result[1] += 1;
            middle = result[1];
        }
    } else {
        middle = min;
        min = result[1];
    }
    std::uniform_int_distribution<size_t> third_dist{0, POPULATION_SIZE - 4};
    result[2] = third_dist(gen);
    if (result[2] >= min) {
        if (result[2] + 1 >= middle) {
            if (result[2] + 2 >= max) {
                result[2] += 3;
            } else {
                result[2] += 2;
            }
        } else {
            result[2] += 1;
        }
    }
    return result;
}

template <typename Generator>
void crossover_weights(Weights& y, const Weights& a, const Weights& b, const Weights& c, Generator& gen) {
    std::uniform_real_distribution<float> unit_dist{0, 1};
    for (size_t i=0; i < PACKS; i++) {
        for (size_t j=0; j < PACK_SIZE; j++) {
            if (unit_dist(gen) < CROSSOVER_RATE) y[i][j] = std::min(std::max(a[i][j] + DIFFERENTIAL_VOLATILITY * (b[i][j] - c[i][j]), MIN_WEIGHT), MAX_WEIGHT);
        }
    }
}

Variables optimize_variables(const float temperature, const std::vector<Pick>& picks, const size_t num_generations,
                             const std::shared_ptr<const Constants>& constants,
                             const std::shared_ptr<const Variables>& initial_variables, const size_t seed) {
    std::cout << "Beginning optimize_variables with population size of " << POPULATION_SIZE << " and " << picks.size()
              << " picks, sampling " << PICKS_PER_GENERATION << " per generation" << std::endl << std::endl;
    std::mt19937_64 gen{seed};
    std::uniform_real_distribution<float> unit_dist{0, 1};
    std::uniform_int_distribution<size_t> pick_selector{0, picks.size() - 1};
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
    std::array<std::pair<size_t, double>, POPULATION_SIZE> indexed_losses;
    std::array<std::pair<size_t, double>, POPULATION_SIZE> indexed_accuracies;
    std::vector<Pick> chosen_picks(PICKS_PER_GENERATION);
    for (size_t i=0; i < PICKS_PER_GENERATION; i++) chosen_picks[i] = picks[pick_selector(gen)];
    std::array<std::array<double, 4>, POPULATION_SIZE> old_losses = run_simulations(population, chosen_picks, temperature, constants);
    std::cout << "Calculated original losses" << std::endl;
    std::vector<Variables> new_population(POPULATION_SIZE);
    for (size_t generation=0; generation < num_generations; generation++) {
        const auto start = std::chrono::high_resolution_clock::now();
        for (size_t i=0; i < PICKS_PER_GENERATION; i++) chosen_picks[i] = picks[pick_selector(gen)];
        for (size_t x=0; x < POPULATION_SIZE; x++) {
            Variables y = population[x];
            std::array<size_t, 3> abc = pick_distinct(x, gen);
            const Variables& a = population[abc[0]];
            const Variables& b = population[abc[1]];
            const Variables& c = population[abc[2]];
            crossover_weights(y.rating_weights, a.rating_weights, b.rating_weights, c.rating_weights, gen);
            crossover_weights(y.pick_synergy_weights, a.pick_synergy_weights, b.pick_synergy_weights, c.pick_synergy_weights, gen);
            crossover_weights(y.fixing_weights, a.fixing_weights, b.fixing_weights, c.fixing_weights, gen);
            crossover_weights(y.internal_synergy_weights, a.internal_synergy_weights, b.internal_synergy_weights, c.internal_synergy_weights, gen);
            crossover_weights(y.openness_weights, a.openness_weights, b.openness_weights, c.openness_weights, gen);
            crossover_weights(y.colors_weights, a.colors_weights, b.colors_weights, c.colors_weights, gen);
            for (size_t i=0; i < NUM_CARDS; i++) {
                if (unit_dist(gen) < CROSSOVER_RATE) y.ratings[i] = std::min(std::max(a.ratings[i] + DIFFERENTIAL_VOLATILITY * (b.ratings[i] - c.ratings[i]), 0.f), MAX_SCORE);
            }
            if (unit_dist(gen) < CROSSOVER_RATE) y.prob_to_include = std::min(std::max(a.prob_to_include + DIFFERENTIAL_VOLATILITY * (b.prob_to_include - c.prob_to_include), 0.f), 0.99f);
            y.prob_multiplier = 1 / (1 - y.prob_to_include);
            if (unit_dist(gen) < CROSSOVER_RATE) y.similarity_clip = std::min(std::max(a.similarity_clip + DIFFERENTIAL_VOLATILITY * (b.similarity_clip - c.similarity_clip), 0.f), 0.99f);
            y.similarity_multiplier = 1 / (1 - y.similarity_clip);
            if (unit_dist(gen) < CROSSOVER_RATE) y.is_fetch_multiplier = std::min(std::max(a.is_fetch_multiplier + DIFFERENTIAL_VOLATILITY * (b.is_fetch_multiplier - c.is_fetch_multiplier), 0.f), 1.f);
            if (unit_dist(gen) < CROSSOVER_RATE) y.has_basic_types_multiplier = std::min(std::max(a.has_basic_types_multiplier + DIFFERENTIAL_VOLATILITY * (b.has_basic_types_multiplier - c.has_basic_types_multiplier), 0.f), 1.f);
            if (unit_dist(gen) < CROSSOVER_RATE) y.is_regular_land_multiplier = std::min(std::max(a.is_regular_land_multiplier + DIFFERENTIAL_VOLATILITY * (b.is_regular_land_multiplier - c.is_regular_land_multiplier), 0.f), 1.f);
            if (unit_dist(gen) < CROSSOVER_RATE) y.equal_cards_synergy = std::min(std::max(a.equal_cards_synergy + DIFFERENTIAL_VOLATILITY * (b.equal_cards_synergy - c.equal_cards_synergy), 0.f), MAX_SCORE);
            new_population[x] = y;
        }
        std::array<std::array<double, 4>, POPULATION_SIZE> losses = run_simulations(new_population, chosen_picks, temperature, constants);
        auto output_result = [=](const std::array<double, 4> &arr) {
            std::cout << "Loss: " << std::setw(WIDTH) << arr[0]
                      << " Categorical Cross-Entropy Loss: " << std::setw(WIDTH) << arr[1]
                      << " Negative Log Accuracy Loss: " << std::setw(WIDTH) << arr[2]
                      << " Accuracy Metric: " << std::setw(WIDTH) << arr[3]
                      << std::endl;
        };
        for (size_t i=0; i < POPULATION_SIZE; i++) {
            if (losses[i][0] < old_losses[i][0]) {
                std::cout << "Moving to ";
                output_result(losses[i]);
                std::cout << "From      ";
                output_result(old_losses[i]);
                population[i] = new_population[i];
                old_losses[i] = losses[i];
            } else {
                std::cout << "Staying with old       ";
                output_result(old_losses[i]);
                std::cout << "Instead of changing to ";
                output_result(losses[i]);
            }
            indexed_losses[i] = {i, old_losses[i][0]};
            indexed_accuracies[i] = {i, old_losses[i][2]};
        }
        auto sort_fn = [](const auto& pair1, const auto& pair2){
            if (std::isnan(pair1.second)) return false;
            else if (std::isnan(pair2.second)) return true;
            else return pair1.second < pair2.second;
        };
        std::sort(indexed_losses.begin(), indexed_losses.end(), sort_fn);
        std::sort(indexed_accuracies.begin(), indexed_accuracies.end(), sort_fn);
        auto total_metrics = std::accumulate(old_losses.begin(), old_losses.end(), std::array<double, 4>{0, 0, 0, 0},
                                             [](const auto &arr1, const auto &arr2) -> std::array<double, 4> {
                                                 return {arr1[0] + arr2[0], arr1[1] + arr2[1], arr1[2] + arr2[2],
                                                         arr1[3] + arr2[3]};
                                             });

        std::cout << "Generation " << generation << std::endl;
        std::ostringstream out_file_name;
        out_file_name << "output/variables-" << generation << ".json";
        save_variables(population[indexed_losses[0].first], out_file_name.str());
        std::cout << "Best Loss:      ";
        output_result(old_losses[indexed_losses[0].first]);
        std::cout << "Best Accuracy:  ";
        output_result(old_losses[indexed_accuracies[0].first]);
        std::cout << "Average:        ";
        output_result({total_metrics[0] / POPULATION_SIZE, total_metrics[1] / POPULATION_SIZE,
                       total_metrics[2] / POPULATION_SIZE, total_metrics[3] / POPULATION_SIZE});
        std::cout << "Median Loss:    ";
        std::array<double, 4> med_low = old_losses[indexed_losses[POPULATION_SIZE / 2 - 1].first];
        std::array<double, 4> med_hi = old_losses[indexed_losses[POPULATION_SIZE / 2].first];
        output_result({(med_low[0] + med_hi[0]) / 2, (med_low[1] + med_hi[1]) / 2,
                       (med_low[2] + med_hi[2]) / 2, (med_low[3] + med_hi[3]) / 2});
        std::cout << "Worst Loss:     ";
        output_result(old_losses[indexed_losses[POPULATION_SIZE - 1].first]);
        const auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        std::cout << "Generation took " << diff.count() << " seconds" << std::endl << std::endl;
    }
    return population[0];
}
