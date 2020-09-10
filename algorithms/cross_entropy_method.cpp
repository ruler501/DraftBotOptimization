//
// Created by Devon Richards on 9/4/2020.
//
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <vector>
#include <chrono>

#include "../draftbot_optimization.h"
#include "shared/parameters.h"

struct NormalDistParameters {
    float mean;
    float std_dev;

    template <typename Generator>
    float operator()(Generator& gen) const {
        std::normal_distribution<float> dist(mean, std_dev);
        return dist(gen);
    }
};

struct WeightsNormalDistParameters {
    std::array<std::array<NormalDistParameters, PACK_SIZE>, PACKS> parameters{};

    WeightsNormalDistParameters() {
        for(size_t i=0; i < PACKS; i++) {
            for (size_t j=0; j < PACK_SIZE; j++) {
                parameters[i][j] = {INITIAL_WEIGHT_MEAN, INITIAL_WEIGHT_STDDEV};
            }
        }
    }

    template <typename Generator>
    Weights operator()(Generator& gen) const {
        Weights result;
        for (size_t i=0; i < PACKS; i++) {
            for (size_t j=0; j < PACK_SIZE; j++) {
                result[i][j] = std::min(std::max(parameters[i][j](gen), MIN_WEIGHT), MAX_WEIGHT);
            }
        }
        return result;
    }

    template<typename Callable>
    Weights extract_parameter(Callable extraction_func) const {
        Weights result;
        for (size_t i=0; i < PACKS; i++) {
            for (size_t j=0; j < PACK_SIZE; j++) {
                result[i][j] = extraction_func(parameters[i][j]);
            }
        }
        return result;
    }

    template<typename Callable>
    void assign_parameters(const Weights& weights, Callable assignment_function) {
        for (size_t i=0; i < PACKS; i++) {
            for (size_t j=0; j < PACK_SIZE; j++) {
                assignment_function(parameters[i][j], weights[i][j]);
            }
        }
    }
};

struct VariablesNormalDistParameters {
    WeightsNormalDistParameters rating_weights_parameters;
    WeightsNormalDistParameters colors_weights_parameters;
    WeightsNormalDistParameters fixing_weights_parameters;
    WeightsNormalDistParameters internal_synergy_weights_parameters;
    WeightsNormalDistParameters pick_synergy_weights_parameters;
    WeightsNormalDistParameters openness_weights_parameters;
#ifdef OPTIMIZE_RATINGS
    std::array<NormalDistParameters, NUM_CARDS> ratings_parameters{};
#endif
    NormalDistParameters prob_to_include_parameters{INITIAL_UNIT_MEAN, INITIAL_UNIT_STDDEV};
    NormalDistParameters  similarity_clip_parameters{INITIAL_UNIT_MEAN, INITIAL_UNIT_STDDEV};
    NormalDistParameters is_fetch_multiplier_parameters{INITIAL_UNIT_MEAN, INITIAL_UNIT_STDDEV};
    NormalDistParameters has_basic_types_multiplier_parameters{INITIAL_UNIT_MEAN, INITIAL_UNIT_STDDEV};
    NormalDistParameters is_regular_land_multiplier_parameters{INITIAL_UNIT_MEAN, INITIAL_UNIT_STDDEV};
    NormalDistParameters equal_cards_synergy_parameters{INITIAL_RATING_MEAN, INITIAL_RATING_STDDEV};

#ifdef OPTIMIZE_RATINGS
    VariablesNormalDistParameters() {
        for (size_t i=0; i < NUM_CARDS; i++) ratings_parameters[i] = {INITIAL_RATING_MEAN, INITIAL_RATING_STDDEV};
    }
#endif

    template <typename Generator>
    Variables operator()(Generator& gen) {
        Variables result;
        result.rating_weights = rating_weights_parameters(gen);
        result.colors_weights = colors_weights_parameters(gen);
        result.fixing_weights = fixing_weights_parameters(gen);
        result.internal_synergy_weights = internal_synergy_weights_parameters(gen);
        result.pick_synergy_weights = pick_synergy_weights_parameters(gen);
        result.openness_weights = openness_weights_parameters(gen);
#ifdef OPTIMIZE_RATINGS
        for (size_t i=0; i < NUM_CARDS; i++) result.ratings[i] = std::min(std::max(ratings_parameters[i](gen), 0.f), MAX_SCORE);
#endif
        result.prob_to_include = std::min(std::max(prob_to_include_parameters(gen), 0.f), 0.99f);
        result.prob_multiplier = 1 / (1 - result.prob_to_include);
        result.similarity_clip = std::min(std::max(similarity_clip_parameters(gen), 0.f), 0.99f);
        result.similarity_multiplier = 1 / (1 - result.similarity_clip);
        result.is_fetch_multiplier = std::min(std::max(is_fetch_multiplier_parameters(gen), 0.f), 1.f);
        result.has_basic_types_multiplier = std::min(std::max(has_basic_types_multiplier_parameters(gen), 0.f), 1.f);
        result.is_regular_land_multiplier = std::min(std::max(is_regular_land_multiplier_parameters(gen), 0.f), 1.f);
        result.equal_cards_synergy = std::min(std::max(equal_cards_synergy_parameters(gen), 0.f), MAX_SCORE);
        return result;
    }
    
    template <typename Callable>
    Variables extract_parameter(Callable retrieval_function) {
        Variables result;
        result.rating_weights = rating_weights_parameters.extract_parameter(retrieval_function);
        result.fixing_weights = fixing_weights_parameters.extract_parameter(retrieval_function);
        result.pick_synergy_weights = pick_synergy_weights_parameters.extract_parameter(retrieval_function);
        result.internal_synergy_weights = internal_synergy_weights_parameters.extract_parameter(retrieval_function);
        result.openness_weights = openness_weights_parameters.extract_parameter(retrieval_function);
        result.colors_weights = colors_weights_parameters.extract_parameter(retrieval_function);
#ifdef OPTIMIZE_RATINGS
        for (size_t i=0; i < NUM_CARDS; i++) result.ratings[i] = retrieval_function(ratings_parameters[i]);
#endif
        result.similarity_clip = retrieval_function(similarity_clip_parameters);
        result.similarity_multiplier = 1 / (1 - result.similarity_clip);
        result.prob_to_include = retrieval_function(prob_to_include_parameters);
        result.prob_multiplier = 1 / (1 - result.prob_multiplier);
        result.is_fetch_multiplier = retrieval_function(is_fetch_multiplier_parameters);
        result.has_basic_types_multiplier = retrieval_function(has_basic_types_multiplier_parameters);
        result.is_regular_land_multiplier = retrieval_function(is_regular_land_multiplier_parameters);
        result.equal_cards_synergy = retrieval_function(equal_cards_synergy_parameters);
        return result;
    }

    Variables extract_means() {
        return extract_parameter([](const NormalDistParameters& params) { return params.mean; });
    }

    Variables extract_std_devs() {
        return extract_parameter([](const NormalDistParameters& params) { return params.std_dev; });
    }

    template <typename Callable>
    void assign_parameters(const Variables& means, Callable assignment_function) {
        rating_weights_parameters.assign_parameters(means.rating_weights, assignment_function);
        pick_synergy_weights_parameters.assign_parameters(means.pick_synergy_weights, assignment_function);
        fixing_weights_parameters.assign_parameters(means.fixing_weights, assignment_function);
        internal_synergy_weights_parameters.assign_parameters(means.internal_synergy_weights, assignment_function);
        openness_weights_parameters.assign_parameters(means.openness_weights, assignment_function);
        colors_weights_parameters.assign_parameters(means.colors_weights, assignment_function);
#ifdef OPTIMIZE_RATINGS
        for (size_t i=0; i < NUM_CARDS; i++) assignment_function(ratings_parameters[i], means.ratings[i]);
#endif
        assignment_function(prob_to_include_parameters, means.prob_to_include);
        assignment_function(similarity_clip_parameters, means.similarity_clip);
        assignment_function(is_fetch_multiplier_parameters, means.is_fetch_multiplier);
        assignment_function(has_basic_types_multiplier_parameters, means.has_basic_types_multiplier);
        assignment_function(is_regular_land_multiplier_parameters, means.is_regular_land_multiplier);
        assignment_function(equal_cards_synergy_parameters, means.equal_cards_synergy);
    }

    void assign_means(const Variables& means) {
        assign_parameters(means, [](NormalDistParameters& params, const float value) { params.mean = value; });
    }

    void assign_std_devs(const Variables& means) {
        assign_parameters(means, [](NormalDistParameters& params, const float value) { params.mean = value; });
    }
};

Weights zeroed_weights() {
    Weights result;
    for (size_t i=0; i < PACKS; i++) {
        for (size_t j=0; j < PACK_SIZE; j++) {
            result[i][j] = 0;
        }
    }
    return result;
}

Variables zeroed_variables() {
    Variables result;
    result.rating_weights = zeroed_weights();
    result.colors_weights = zeroed_weights();
    result.fixing_weights = zeroed_weights();
    result.pick_synergy_weights = zeroed_weights();
    result.internal_synergy_weights = zeroed_weights();
    result.openness_weights = zeroed_weights();
#ifdef OPTIMIZE_RATINGS
    for (size_t i=0; i < NUM_CARDS; i++) result.ratings[i] = 0;
#endif
    result.similarity_clip = 0;
    result.similarity_multiplier = 1;
    result.prob_to_include = 0;
    result.prob_multiplier = 1;
    result.is_fetch_multiplier = 0;
    result.has_basic_types_multiplier = 0;
    result.is_regular_land_multiplier = 0;
    result.equal_cards_synergy = 0;
    return result;
}

void average_weights(Weights& target, const Weights& source) {
    for (size_t i=0; i < PACKS; i++) {
        for (size_t j=0; j < PACK_SIZE; j++) {
            target[i][j] += source[i][j] / KEEP_BEST;
        }
    }
}

void variance_weights(Weights& target, const Weights& source, const Weights& mean) {
    for (size_t i=0; i < PACKS; i++) {
        for (size_t j=0; j < PACK_SIZE; j++) {
            target[i][j] += std::pow(source[i][j] - mean[i][j], 2) / KEEP_BEST;
        }
    }
}

WeightsNormalDistParameters make_weight_dist(const Weights& mean, const Weights& variance) {
    WeightsNormalDistParameters result;
    for (size_t i=0; i < PACKS; i++) {
        for (size_t j=0; j < PACK_SIZE; j++) {
            result.parameters[i][j] = {mean[i][j], std::sqrt(variance[i][j])};
        }
    }
    return result;
}

Variables optimize_variables(const float temperature, const std::vector<Pick>& picks, const size_t num_generations,
                             const std::shared_ptr<const Constants>& constants,
                             const std::shared_ptr<const Variables>& initial_variables, const size_t seed) {
    std::mt19937_64 gen(seed);
    VariablesNormalDistParameters dist;
    if (initial_variables) {
        dist.assign_means(*initial_variables);
        dist.assign_std_devs(load_variables("output/load-std-devs.json"));
    }
    std::vector<Variables> population(POPULATION_SIZE);
    std::uniform_int_distribution<size_t> pick_selector(0, picks.size() - 1);
    std::vector<Pick> chosen_picks(PICKS_PER_GENERATION);
    std::array<std::pair<size_t, double>, POPULATION_SIZE> indexed_losses;
    for (size_t generation=0; generation < num_generations; generation++) {
        const auto start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < POPULATION_SIZE; i++) population[i] = dist(gen);
        for (size_t i = 0; i < PICKS_PER_GENERATION; i++) chosen_picks[i] = picks[pick_selector(gen)];
        std::array<std::array<double, 4>, POPULATION_SIZE> losses = run_simulations(population, chosen_picks,
                                                                                    temperature, constants);
        std::array<std::pair<std::size_t, double>, POPULATION_SIZE> indexed_accuracies;
        for (size_t i = 0; i < POPULATION_SIZE; i++) {
            indexed_losses[i] = {i, losses[i][0]};
            indexed_accuracies[i] = {i, losses[i][2]};
        }
        auto sort_fn = [](const auto &pair1, const auto &pair2) {
            if (std::isnan(pair1.second)) return false;
            else if (std::isnan(pair2.second)) return true;
            else return pair1.second < pair2.second;
        };
        std::sort(indexed_losses.begin(), indexed_losses.end(), sort_fn);
        std::sort(indexed_accuracies.begin(), indexed_accuracies.end(), sort_fn);
        auto total_metrics = std::accumulate(losses.begin(), losses.end(), std::array<double, 4>{0, 0, 0, 0},
                                             [](const auto &arr1, const auto &arr2) -> std::array<double, 4> {
                                                 return {arr1[0] + arr2[0], arr1[1] + arr2[1], arr1[2] + arr2[2],
                                                         arr1[3] + arr2[3]};
                                             });
        Variables sum_values = zeroed_variables();
        for (size_t i = 0; i < KEEP_BEST; i++) {
            const Variables &variables = population[indexed_accuracies[i].first];
            average_weights(sum_values.rating_weights, variables.rating_weights);
            average_weights(sum_values.fixing_weights, variables.fixing_weights);
            average_weights(sum_values.pick_synergy_weights, variables.pick_synergy_weights);
            average_weights(sum_values.internal_synergy_weights, variables.internal_synergy_weights);
            average_weights(sum_values.colors_weights, variables.colors_weights);
            average_weights(sum_values.openness_weights, variables.openness_weights);
#ifdef OPTIMIZE_RATINGS
            for (size_t j = 0; j < NUM_CARDS; j++) sum_values.ratings[j] += variables.ratings[j] / KEEP_BEST;
#endif
            sum_values.prob_to_include += variables.prob_to_include / KEEP_BEST;
            sum_values.similarity_clip += variables.similarity_clip / KEEP_BEST;
            sum_values.is_fetch_multiplier += variables.is_fetch_multiplier / KEEP_BEST;
            sum_values.has_basic_types_multiplier += variables.has_basic_types_multiplier / KEEP_BEST;
            sum_values.is_regular_land_multiplier += variables.is_regular_land_multiplier / KEEP_BEST;
            sum_values.equal_cards_synergy += variables.equal_cards_synergy / KEEP_BEST;
        }
        Variables var_values = zeroed_variables();
        for (size_t i = 0; i < KEEP_BEST; i++) {
            const Variables &variables = population[indexed_accuracies[i].first];
            variance_weights(var_values.rating_weights, variables.rating_weights, sum_values.rating_weights);
            variance_weights(var_values.fixing_weights, variables.fixing_weights, sum_values.fixing_weights);
            variance_weights(var_values.pick_synergy_weights, variables.pick_synergy_weights, sum_values.pick_synergy_weights);
            variance_weights(var_values.internal_synergy_weights, variables.internal_synergy_weights, sum_values.internal_synergy_weights);
            variance_weights(var_values.colors_weights, variables.colors_weights, sum_values.colors_weights);
            variance_weights(var_values.openness_weights, variables.openness_weights, sum_values.openness_weights);
#ifdef OPTIMIZE_RATINGS
            for (size_t j = 0; j < NUM_CARDS; j++) var_values.ratings[j] += std::pow(variables.ratings[j] - sum_values.ratings[j], 2) / KEEP_BEST;
#endif
            var_values.prob_to_include += std::pow(variables.prob_to_include - sum_values.prob_to_include, 2) / KEEP_BEST;
            var_values.similarity_clip += std::pow(variables.similarity_clip - sum_values.similarity_clip, 2) / KEEP_BEST;
            var_values.is_fetch_multiplier += std::pow(variables.is_fetch_multiplier - sum_values.is_fetch_multiplier, 2) / KEEP_BEST;
            var_values.has_basic_types_multiplier += std::pow(variables.has_basic_types_multiplier - sum_values.has_basic_types_multiplier, 2) / KEEP_BEST;
            var_values.is_regular_land_multiplier += std::pow(variables.is_regular_land_multiplier - sum_values.is_regular_land_multiplier, 2) / KEEP_BEST;
            var_values.equal_cards_synergy += std::pow(variables.equal_cards_synergy - sum_values.equal_cards_synergy, 2) / KEEP_BEST;
        }
        dist.rating_weights_parameters = make_weight_dist(sum_values.rating_weights, var_values.rating_weights);
        dist.fixing_weights_parameters = make_weight_dist(sum_values.fixing_weights, var_values.fixing_weights);
        dist.pick_synergy_weights_parameters = make_weight_dist(sum_values.pick_synergy_weights, var_values.pick_synergy_weights);
        dist.internal_synergy_weights_parameters = make_weight_dist(sum_values.internal_synergy_weights, var_values.internal_synergy_weights);
        dist.openness_weights_parameters = make_weight_dist(sum_values.openness_weights, var_values.openness_weights);
        dist.colors_weights_parameters = make_weight_dist(sum_values.colors_weights, var_values.colors_weights);
#ifdef OPTIMIZE_RATINGS
        for (size_t i=0; i < NUM_CARDS; i++) dist.ratings_parameters[i] = {sum_values.ratings[i], std::sqrt(var_values.ratings[i])};
#endif
        dist.prob_to_include_parameters = {sum_values.prob_to_include, std::sqrt(var_values.prob_to_include)};
        dist.similarity_clip_parameters = {sum_values.similarity_clip, std::sqrt(var_values.similarity_clip)};
        dist.is_fetch_multiplier_parameters = {sum_values.is_fetch_multiplier, std::sqrt(var_values.is_fetch_multiplier)};
        dist.has_basic_types_multiplier_parameters = {sum_values.has_basic_types_multiplier, std::sqrt(var_values.has_basic_types_multiplier)};
        dist.is_regular_land_multiplier_parameters = {sum_values.is_regular_land_multiplier, std::sqrt(var_values.is_regular_land_multiplier)};
        dist.equal_cards_synergy_parameters = {sum_values.equal_cards_synergy, std::sqrt(var_values.equal_cards_synergy)};
        auto output_result = [=](const std::array<double, 4> &arr) {
            std::cout << "Loss: " << std::setw(WIDTH) << arr[0]
                      << " Categorical Cross-Entropy Loss: " << std::setw(WIDTH) << arr[1]
                      << " Negative Log Accuracy Loss: " << std::setw(WIDTH) << arr[2]
                      << " Accuracy Metric: " << std::setw(WIDTH) << arr[3]
                      << std::endl;
        };
        std::cout << "Generation " << generation << std::endl;
        std::ostringstream out_file_name;
        out_file_name << "output/variables-" << generation << ".json";
        std::ostringstream out_file_name_means;
        out_file_name_means << "output/means-" << generation << ".json";
        std::ostringstream out_file_name_std_dev;
        out_file_name_std_dev << "output/std-devs-" << generation << ".json";
        save_variables(population[indexed_accuracies[0].first], out_file_name.str());
        save_variables(dist.extract_means(), out_file_name_means.str());
        save_variables(dist.extract_std_devs(), out_file_name_std_dev.str());
        std::cout << "Best Loss:      ";
        output_result(losses[indexed_losses[0].first]);
        std::cout << "Best Accuracy:  ";
        output_result(losses[indexed_accuracies[0].first]);
        std::cout << "Worst Survivor: ";
        output_result(losses[indexed_losses[KEEP_BEST - 1].first]);
        std::cout << "Average:        ";
        output_result({total_metrics[0] / POPULATION_SIZE, total_metrics[1] / POPULATION_SIZE,
                       total_metrics[2] / POPULATION_SIZE, total_metrics[3] / POPULATION_SIZE});
        std::cout << "Median Loss:    ";
        std::array<double, 4> med_low = losses[indexed_accuracies[POPULATION_SIZE / 2 - 1].first];
        std::array<double, 4> med_hi = losses[indexed_accuracies[POPULATION_SIZE / 2].first];
        output_result({(med_low[0] + med_hi[0]) / 2, (med_low[1] + med_hi[1]) / 2,
                       (med_low[2] + med_hi[2]) / 2, (med_low[3] + med_hi[3]) / 2});
        std::cout << "Worst Loss:     ";
        output_result(losses[indexed_accuracies[POPULATION_SIZE - 1].first]);
        const auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        std::cout << "Generation took " << diff.count() << " seconds" << std::endl << std::endl;
    }
    return dist.extract_means();
}