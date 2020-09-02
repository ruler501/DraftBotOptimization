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
#include <random>
#include <thread>
#include <vector>

#ifdef USE_SYCL
#include <CL/sycl.hpp>
#include <numeric>

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

#include "draftbot_optimization.h"

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

float get_prob_to_cast(size_t cmc, size_t required_a, size_t required_b, size_t land_count_a,
                       size_t land_count_b, size_t land_count_ab, const Constants& constants) {
    cmc = std::min(cmc, PROB_DIM_0 - 1);
    required_a = std::min(required_a, PROB_DIM_1 - 1);
    required_b = std::min(required_b, PROB_DIM_2 - 1);
    land_count_a = std::min(land_count_a, PROB_DIM_3 - 1);
    land_count_b = std::min(land_count_b, PROB_DIM_4 - 1);
    land_count_ab = std::min(land_count_ab, PROB_DIM_5 - 1);
    return constants.prob_to_cast[cmc][required_a][required_b][land_count_a][land_count_b][land_count_ab];
}

float get_casting_probability(const Lands& lands, const size_t card_index, const Constants& constants) {
    const ColorRequirement &color_requirement = constants.color_requirements[card_index];
    size_t num_requirements = 5;
    for (size_t i = 0; i < color_requirement.size(); i++) {
        if (color_requirement[i].second == 0) {
            num_requirements = i;
            break;
        }
    }
    if (num_requirements == 0) {
        return 1;
    } else if (num_requirements < 3) {
        const std::pair<std::array<bool, 32>, size_t>& first_requirement = color_requirement[0];
        std::array<bool, 32> color_a = first_requirement.first;
        size_t required_a = first_requirement.second;
        std::array<bool, 32> color_b{false};
        size_t required_b = 0;
        if (num_requirements == 2) {
            const std::pair<std::array<bool, 32>, size_t>& second_requirement = color_requirement[1];
            color_b = second_requirement.first;
            required_b = second_requirement.second;
        }
        if (required_a < required_b) {
            const size_t temp = required_a;
            required_a = required_b;
            required_b = temp;
            std::array<bool, 32> temp_colors = color_a;
            color_a = color_b;
            color_b = temp_colors;
        }
        size_t land_count_a = 0;
        size_t land_count_b = 0;
        size_t land_count_ab = 0;
        for (size_t i=0; i < NUM_COMBINATIONS; i++) {
            const std::pair<Colors, size_t>& entry = lands[i];
            bool intersection_a = color_a[i];
            bool intersection_b = color_b[i];
            if(intersection_a && !intersection_b) {
                land_count_a += entry.second;
            } else if(!intersection_a && intersection_b) {
                land_count_b += entry.second;
#pragma clang diagnostic push
#pragma ide diagnostic ignored "ConstantConditionsOC"
            } else if(intersection_a && intersection_b) {
                land_count_ab += entry.second;
            }
#pragma clang diagnostic pop
        }
        return get_prob_to_cast(constants.cmcs[card_index], required_a, required_b, land_count_a, land_count_b, land_count_ab, constants);
    } else {
        size_t total_devotion = 0;
        float probability = 1.f;
        const size_t cmc = constants.cmcs[card_index];
        for (size_t i=0; i < num_requirements; i++) {
            const std::pair<std::array<bool, 32>, size_t>& entry = color_requirement[i];
            total_devotion += entry.second;
            size_t land_count = 0;
            for (size_t j=0; j < NUM_COMBINATIONS; j++) {
                const std::pair<Colors, size_t>& entry2 = lands[j];
                if (entry.first[j]) {
                    land_count += entry2.second;
                }
            }
            probability *= get_prob_to_cast(cmc, entry.second, 0, land_count, 0, 0, constants);
        }
        size_t land_count = 0;
        for (size_t i=0; i < NUM_COMBINATIONS; i++) {
            const std::pair<Colors, size_t>& entry2 = lands[i];
            for (size_t j=0; j < num_requirements; j++) {
                const std::pair<std::array<bool, 32>, size_t>& entry = color_requirement[j];
                if (entry.first[i]) {
                    land_count += entry2.second;
                    break;
                }
            }
        }
        return probability * get_prob_to_cast(cmc, total_devotion, 0, land_count, 0, 0, constants);
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

float rating_oracle(const size_t card_index, const Lands& lands, const Variables& variables, const Pick& pick, const Constants& constants) {
    const size_t real_card_index = pick.in_pack[card_index];
    return get_casting_probability(lands, real_card_index, constants) * variables.ratings[real_card_index];
}

float pick_synergy_oracle(const size_t card_index, const Lands& lands, const Variables& variables, const Pick& pick,
                          const Constants& constants) {
    size_t num_valid_indices = pick.picked.size();
    for (size_t i=0; i < pick.picked.size(); i++) {
        if (pick.picked[i] == std::numeric_limits<size_t>::max()) {
            num_valid_indices = i;
            break;
        }
    }
    if (num_valid_indices == 0) return 0;
    float total_synergy = 0;
    const Embedding& embedding = constants.embeddings[pick.in_pack[card_index]];
    for (size_t i=0; i < num_valid_indices; i++) {
        const size_t card = pick.picked[i];
        const float probability = get_casting_probability(lands, card, constants);
        if (probability >= variables.prob_to_include) {
            total_synergy += probability * calculate_synergy(embedding, constants.embeddings[card], variables);
        }
    }
    return total_synergy * get_casting_probability(lands, pick.in_pack[card_index], constants) / num_valid_indices;
}

float fixing_oracle(const size_t card_index, const Lands& lands, const Variables& variables, const Pick& pick, const Constants& constants) {
    const size_t card_real_index = pick.in_pack[card_index];
    if (constants.is_land[card_real_index]) {
        float overlap = 0;
        for (size_t i=0; i < NUM_COLORS; i++){
            if (constants.card_colors[card_real_index][i]) {
                size_t count = 0;
                for (size_t j=0; j < NUM_COMBINATIONS; j++) if (lands[j].first[i]) count += lands[j].second;
                if (count > 2) {
                    overlap += 2;
                }
            }
        }
        if (constants.is_fetch[card_real_index]) return overlap;
        else if (constants.has_basic_land_types[card_real_index]) return 0.75f * overlap;
        else return 0.5f * overlap;
    } else return 0;
}

float internal_synergy_oracle(const size_t, const Lands& lands, const Variables& variables, const Pick& pick, const Constants& constants) {
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
    for (size_t i=0; i < num_valid_indices; i++) probabilities[i] = get_casting_probability(lands, pick.picked[i], constants);
    for(size_t i=0; i < num_valid_indices; i++) {
        const float probability = probabilities[i];
        if (probability >= variables.prob_to_include) {
            const Embedding& embedding = constants.embeddings[pick.picked[i]];
            float card_synergy = 0;
            for (size_t j = 0; j < i; j++) {
                const float probability2 = probabilities[j];
                if (probability2 >= variables.prob_to_include) {
                    card_synergy += probability2 * calculate_synergy(embedding, constants.embeddings[pick.picked[j]], variables);
                }
            }
            internal_synergy += probability * card_synergy;
        }
    }
    return internal_synergy / (float)(num_valid_indices * (num_valid_indices + 1));
}

template<size_t Size>
float sum_gated_rating(const Lands& lands, const Variables& variables, const std::array<size_t, Size>& indices,
                       const Constants& constants) {
    size_t num_valid_indices = indices.size();
    for (size_t i=0; i < indices.size(); i++) {
        if (indices[i] == std::numeric_limits<size_t>::max()) {
            num_valid_indices = i;
            break;
        }
    }
    float result = 0;
    for (size_t i=0; i < num_valid_indices; i++) {
        const size_t index = indices[i];
        const float probability = get_casting_probability(lands, index, constants);
        if (probability >= variables.prob_to_include) {
            result += variables.ratings[index] * probability;
        }
    }
    return result / indices.size();
}

float openness_oracle(const size_t, const Lands& lands, const Variables& variables, const Pick& pick, const Constants& constants) {
    return sum_gated_rating(lands, variables, pick.seen, constants);
}

float colors_oracle(const size_t, const Lands& lands, const Variables& variables, const Pick& pick, const Constants& constants) {
    return sum_gated_rating(lands, variables, pick.picked, constants);
}

float get_score(const size_t card_index, const Lands& lands, const Variables& variables, const Pick& pick,
                const float rating_weight, const float pick_synergy_weight, const float fixing_weight,
                const float internal_synergy_weight, const float openness_weight, const float colors_weight,
                const Constants& constants) {
    const float rating_score = rating_oracle(card_index, lands, variables, pick, constants);
//    return rating_score*rating_weight;
//     std::cout << rating_score << "*" << rating_weight;
    const float pick_synergy_score = pick_synergy_oracle(card_index, lands, variables, pick, constants);
//    return pick_synergy_score*pick_synergy_weight;
//     std::cout << " + " << pick_synergy_score << "*" << pick_synergy_weight;
    const float fixing_score = fixing_oracle(card_index, lands, variables, pick, constants);
//    return fixing_score*fixing_weight;
//     std::cout << " + " << fixing_score << "*" << fixing_weight;
    const float internal_synergy_score = internal_synergy_oracle(card_index, lands, variables, pick, constants);
//    return internal_synergy_score*internal_synergy_weight;
//     std::cout << " + " << internal_synergy_score << "*" << internal_synergy_weight;
    const float openness_score = openness_oracle(card_index, lands, variables, pick, constants);
//    return openness_score*openness_weight;
//     std::cout << " + " << openness_score << "*" << openness_weight;
    const float colors_score = colors_oracle(card_index, lands, variables, pick, constants);
//    return colors_score*colors_weight;
//     std::cout << " + " << colors_score << "*" << colors_weight << std::endl;
    return rating_score*rating_weight + pick_synergy_score*pick_synergy_weight + fixing_score*fixing_weight
           + internal_synergy_score*internal_synergy_weight + openness_score*openness_weight + colors_score*colors_weight;
}

float do_climb(const size_t card_index, const Variables& variables, const Pick& pick, const Constants& constants) {
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
                    if (add_index == remove_index) continue;
                    Lands new_lands = lands;
                    new_lands[remove_index].second -= 1;
                    new_lands[add_index].second += 1;
                    float score = get_score(card_index, new_lands, variables, pick, rating_weight, pick_synergy_weight,
                                            fixing_weight, internal_synergy_weight, openness_weight, colors_weight,
                                            constants);
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

std::pair<double, bool> calculate_loss(const Pick& pick, const Variables& variables, const float temperature, const Constants& constants) {
    std::array<double, MAX_PACK_SIZE> scores{0};
    std::array<double, MAX_PACK_SIZE> softmaxed{0};
    double denominator = 0;
    size_t num_valid_indices = pick.in_pack.size();
    for (size_t i=0; i < pick.in_pack.size(); i++) {
        if (pick.in_pack[i] == std::numeric_limits<size_t>::max()) {
            num_valid_indices = i;
            break;
        }
    }
    for (size_t i=0; i < num_valid_indices; i++) {
        scores[i] = EXP((double) do_climb(i, variables, pick, constants) / temperature);
        denominator += scores[i];
    }
    double max_score = 0;
    for (size_t i=0; i < num_valid_indices; i++) {
        softmaxed[i] = scores[i] / denominator;
        max_score = std::max(max_score, softmaxed[i]);
    }
    for(size_t i=0; i < num_valid_indices; i++) {
        if (pick.in_pack[i] == pick.chosen_card) {
            if (softmaxed[i] >= 0) {
                return {-LOG(softmaxed[i]), softmaxed[i] == max_score};
            } else {
                return {-1, false};
            }
        }
    }
    return {-2, false};
}

std::array<double, 2> get_batch_loss(const std::array<Pick, PICKS_PER_GENERATION> picks, const Variables& variables,
                                     const float temperature, const Constants& constants) {
    double sum_loss = 0;
    size_t count_correct = 0;
    for (size_t i=0; i < PICKS_PER_GENERATION; i++) {
        const std::pair<double, bool> pick_loss = calculate_loss(picks[i], variables, temperature, constants);
        if (pick_loss.first < 0) return {-1, -1};
        sum_loss += pick_loss.first;
        if (pick_loss.second) count_correct++;
    }
    return {sum_loss / PICKS_PER_GENERATION, count_correct / (double)PICKS_PER_GENERATION};
}

class calculate_batch_loss;

std::array<std::array<double, 2>, POPULATION_SIZE> run_simulations(const std::vector<Variables>& variables,
                                                                   const std::vector<Pick>& picks, const float temperature,
                                                                   const std::shared_ptr<const Constants>& constants) {
    std::array<std::array<double, 2>, POPULATION_SIZE> results{{0,0}};
#ifdef USE_SYCL
    using namespace cl::sycl;
    try {
        const auto dev_type = cl::sycl::info::device_type::gpu;
        // Platform selection
        auto plats = cl::sycl::platform::get_platforms();
        if (plats.empty()) throw std::runtime_error{ "No OpenCL platform found." };

        auto plat = plats.at(1);
        // Device selection
        auto devs = plat.get_devices(dev_type);
        if (devs.empty()) throw std::runtime_error{ "No OpenCL device of specified type found on selected platform." };

        auto dev = devs.at(0);

        // Context, queue, buffer creation
        auto async_error_handler = [](const cl::sycl::exception_list& errors) { for (const auto& error : errors) throw error; }; // NOLINT(misc-throw-by-value-catch-by-reference,hicpp-exception-baseclass)

        cl::sycl::context ctx{ dev, async_error_handler };

        cl::sycl::gpu_selector selector;

        cl::sycl::queue myQueue{selector};

        /* Create a scope to control data synchronisation of buffer objects. */
        {
            buffer<Variables, 1> inputBuffer(variables.data(), sycl::range<1>(POPULATION_SIZE));
            buffer<Pick, 1> picksBuffer(picks.data(), sycl::range<1>(picks.size()));
            buffer<Constants, 1> constantsBuffer(constants.get(), sycl::range<1>(1));
            buffer<std::pair<double, bool>, 2> outputBuffer{{POPULATION_SIZE, PICKS_PER_GENERATION}};

            /* Submit a command_group to execute from the queue. */
            myQueue.submit([&](handler &cgh) {
                /* Create accessors for accessing the input and output data within the
                 * kernel. */
                auto inputPtr = inputBuffer.get_access<access::mode::read>(cgh);
                auto pickPtr = picksBuffer.get_access<access::mode::read>(cgh);
                auto constantsPtr = constantsBuffer.get_access<access::mode::read>(cgh);
                auto outputPtr = outputBuffer.get_access<access::mode::write>(cgh);

                cgh.parallel_for<class calculate_batch_loss>(sycl::range<2>(POPULATION_SIZE, picks.size()),
                                                             [=](cl::sycl::id<2> wiID) {
                                                                const Variables& variables = inputPtr[wiID[0]];
                                                                const Pick& pick = pickPtr[wiID[1]];
                                                                const Constants& constants = constantsPtr[0];
                                                                outputPtr[wiID] = calculate_loss(pick, variables, temperature, constants);
                                                             }
                );
            });
            myQueue.wait();
            auto readOutputPtr = outputBuffer.get_access<access::mode::read>();
            for (size_t i=0; i < POPULATION_SIZE; i++) {
                size_t count_correct = 0;
                for (size_t j=0; j < PICKS_PER_GENERATION; j++) {
//                    if (j % 1000 == 0 || readOutputPtr[i][j].first < 0 || ISINF(readOutputPtr[i][j].first) || j == 10606 || j == 10664 || j==11782 || j==11782) std::cout << i << "," << j << " " << readOutputPtr[i][j].first << "," << readOutputPtr[i][j].second << std::endl;
                    results[i][0] += readOutputPtr[i][j].first;
                    if (readOutputPtr[i][j].second) count_correct++;
                }
                results[i][1] = count_correct / (double)PICKS_PER_GENERATION;
                results[i][0] /= results[i][1] * PICKS_PER_GENERATION;
//                std::cout << "Generation " << i << ": " << results[i][0] << ", " << results[i][1] << ", " << 1 / results[i][1] << std::endl;
            }
        }
    } catch (exception& e) {
        /* In the case of an exception being throw, print the error message and
         * rethrow it. */
        std::cerr << e.what() << std::endl;
        std::rethrow_exception(std::current_exception());
    }
#else
    std::array<std::future<std::array<double, 2>>, POPULATION_SIZE> futures;
    std::array<std::thread, POPULATION_SIZE> threads;
    for (size_t i=0; i < POPULATION_SIZE; i++) {
        std::packaged_task<std::array<double, 2>()> task(
                [&variables, i, &picks, temperature, &constants] {
                    return get_batch_loss(picks, variables[i], temperature, picks.size(), *constants);
                }); // wrap the function
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

Weights crossover_weights(const Weights& weights1, const Weights& weights2, std::mt19937_64& gen) {
    Weights result = weights1;
    for (size_t pack=0; pack < PACKS; pack++) {
        for (size_t pick=0; pick < PACK_SIZE; pick++) {
            result[pack][pick] = (weights1[pack][pick] + weights2[pack][pick]) / 2;
        }
    }
    return result;
}

Variables crossover_variables(const Variables& variables1, const Variables& variables2, std::mt19937_64& gen) {
    Variables result = variables1;
    result.rating_weights = crossover_weights(variables1.rating_weights, variables2.rating_weights, gen);
    result.pick_synergy_weights = crossover_weights(variables1.pick_synergy_weights, variables2.pick_synergy_weights, gen);
    result.fixing_weights = crossover_weights(variables1.fixing_weights, variables2.fixing_weights, gen);
    result.internal_synergy_weights = crossover_weights(variables1.internal_synergy_weights, variables2.fixing_weights, gen);
    result.openness_weights = crossover_weights(variables1.openness_weights, variables2.openness_weights, gen);
    result.colors_weights = crossover_weights(variables1.colors_weights, variables2.colors_weights, gen);
    result.prob_to_include = (variables1.prob_to_include + variables2.prob_to_include) / 2;
    result.similarity_clip = (variables1.similarity_clip + variables2.similarity_clip) / 2;
    result.similarity_multiplier = 1 / (1 - result.similarity_clip);
    for (size_t i=0; i < NUM_CARDS; i++) result.ratings[i] = (variables1.ratings[i] + variables2.ratings[i]) / 2;
    return result;
}

Weights mutate_weights(Weights& weights, std::mt19937_64& gen) {
    std::normal_distribution<float> std_dev_1{0, 1.0};
    std::uniform_int_distribution<size_t> int_distribution(0, 4);
    for (auto& pack : weights) {
        for (float& weight : pack) {
            if (int_distribution(gen) == 0) weight += std_dev_1(gen);
        }
    }
    return weights;
}

Variables mutate_variables(Variables& variables, std::mt19937_64& gen) {
    std::normal_distribution<float> std_dev_005{0, 0.05f};
    std::normal_distribution<float> std_dev_05{0, 0.5f};
    std::uniform_int_distribution<size_t> int_distribution(0, 4);
    std::uniform_int_distribution<size_t> int_distribution2(0, 9);
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
    for (size_t i=0; i < NUM_CARDS; i++) {
        if(int_distribution2(gen) == 0) {
            variables.ratings[i] += std_dev_05(gen);
            variables.ratings[i] = std::max(std::min(10.f, variables.ratings[i]), 0.f);
        }
    }
    return variables;
}

Variables optimize_variables(const float temperature, const std::vector<Pick>& picks, const size_t num_generations,
                             const std::shared_ptr<const Constants>& constants) {
    std::cout << "Beginning optimize_variables with population size of " << POPULATION_SIZE << std::endl << std::endl;
    std::random_device rd{};
    std::mt19937_64 gen{rd()};
    std::uniform_int_distribution<size_t> crossover_selector(0, POPULATION_SIZE / 2 + POPULATION_SIZE % 2 - 1);
    std::uniform_int_distribution<size_t> pick_selector(0, picks.size() - 1);
    std::vector<Variables> population(POPULATION_SIZE);
    std::vector<Pick> chosen_picks(PICKS_PER_GENERATION);
    for (size_t i=0; i < POPULATION_SIZE; i++) population[i].ratings = INITIAL_RATINGS;
    for(size_t generation=0; generation < num_generations; generation++) {
        const auto start = std::chrono::high_resolution_clock::now();
//        std::cout << "Beginning crossover" << std::endl;
        for (size_t i=0; i < POPULATION_SIZE / 2 + POPULATION_SIZE % 2; i++) {
            size_t index1 = crossover_selector(gen);
            size_t index2 = crossover_selector(gen);
            population[POPULATION_SIZE / 2 + i] = crossover_variables(population[index1], population[index2], gen);
        }
//        std::cout << "Beginning mutation" << std::endl;
        for (Variables& variables : population) {
            mutate_variables(variables, gen);
        }
//        std::cout << "Selecting picks for generation" << std::endl;
        for (size_t i=0; i < PICKS_PER_GENERATION; i++) chosen_picks[i] = picks[pick_selector(gen)];
//        std::cout << "Beginning calculating losses" << std::endl;
        std::array<std::array<double, 2>, POPULATION_SIZE> losses = run_simulations(population, chosen_picks, temperature, constants);
        std::array<std::pair<std::size_t, std::array<double, 2>>, POPULATION_SIZE> indexed_losses;
        for (size_t i=0; i < POPULATION_SIZE; i++) indexed_losses[i] = {i, losses[i]};
        std::sort(indexed_losses.begin(), indexed_losses.end(),
                  [](const auto& pair1, const auto& pair2){ return pair1.second[0] < pair2.second[0]; });
        std::sort(losses.begin(), losses.end(), [](auto& arr1, auto& arr2){ return arr1[1] > arr2[1]; });
        std::array<double, 2> total_metrics = std::accumulate(losses.begin(), losses.end(), std::array<double, 2>{0, 0},
                                                              [](auto& arr1, auto& arr2)->std::array<double, 2>{ return {arr1[0] + arr2[0], arr1[1] + arr2[1]}; });
        std::cout << "Generation: " << generation << std::endl << "Best Loss: " << indexed_losses[0].second[0]
                  << " with Accuracy: " << indexed_losses[0].second[1] << std::endl
                  << "Best Accuracy: " << losses[0][1] << " with Loss: " << losses[0][1] << std::endl
                  << "Average Loss: " << total_metrics[0] / POPULATION_SIZE << " with Average Accuracy: "
                  << total_metrics[1] / POPULATION_SIZE << std::endl;
        std::vector<Variables> temp_population(POPULATION_SIZE);
        for (size_t i=0; i < POPULATION_SIZE; i++) temp_population[i] = population[indexed_losses[i].first];
        population = temp_population;
        const auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        std::cout << "Generation took " << diff.count() << " seconds" << std::endl << std::endl;
    }
    return population[0];
}

int main(const int argc, const char* argv[]) {
    if (argc != 3) {
        return -1;
    }
    std::cout.precision(20);
    const float temperature = std::strtof(argv[1], nullptr);
    const size_t generations = std::strtoull(argv[2], nullptr, 10);
    const std::shared_ptr<const Constants> constants = [](){ // NOLINT(cert-err58-cpp)
        std::shared_ptr<Constants> result_ptr = std::make_shared<Constants>();
        populate_constants("data/intToCard.json", *result_ptr);
        populate_prob_to_cast("data/probTable.json", *result_ptr);
        return result_ptr;
    }();
    const std::vector<Pick> all_picks = load_picks("data/drafts/");
    const Variables best_variables = optimize_variables(temperature, all_picks, generations, constants);
    save_variables(best_variables, "output/variables.json");
}
