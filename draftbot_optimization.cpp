#include <cstddef>
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
#include <iomanip>
#include <iostream>
#include <random>
#include <thread>
#include <vector>
#include <numeric>

#ifdef USE_SYCL
#include <CL/sycl.hpp>
#include <concurrentqueue.h>

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
    unsigned char num_requirements = color_requirement.second;
    if (num_requirements == 0) {
        return 1;
    } else if (num_requirements == 1) {
        const std::pair<std::array<unsigned char, NUM_COMBINATIONS>, unsigned char>& requirement = color_requirement.first[0];
        unsigned char required_a = requirement.second;
        const std::array<unsigned char, NUM_COMBINATIONS>& color_a = requirement.first;
        unsigned char land_count_a = 0;
        for (size_t i=1; i < COLORS.size() + 1; i++) land_count_a += color_a[i] * lands[i].second;
        return get_prob_to_cast(constants.cmcs[card_index], required_a, 0, land_count_a, 0, 0, constants);
    } else if (num_requirements == 2) {
        const std::pair<std::array<unsigned char, NUM_COMBINATIONS>, unsigned char>& first_requirement = color_requirement.first[0];
        const std::array<unsigned char, NUM_COMBINATIONS>& color_a = first_requirement.first;
        unsigned char required_a = first_requirement.second;
        const std::pair<std::array<unsigned char, NUM_COMBINATIONS>, unsigned char>& second_requirement = color_requirement.first[1];
        const std::array<unsigned char, NUM_COMBINATIONS>& color_b = second_requirement.first;
        unsigned char required_b = second_requirement.second;
        unsigned char land_count_a = 0;
        unsigned char land_count_b = 0;
        unsigned char land_count_ab = 0;
        for (size_t i=1; i < COLORS.size() + 1; i++) {
            const unsigned char count = lands[i].second;
            unsigned char intersection_a = color_a[i];
            unsigned char intersection_b = color_b[i];
            land_count_a += intersection_a * (1-intersection_b) * count;
            land_count_b += (1-intersection_a) * intersection_b * count;
            land_count_ab += intersection_a * intersection_b * count;
        }
        if (required_a < required_b) {
            return get_prob_to_cast(constants.cmcs[card_index], required_b, required_a, land_count_b, land_count_a, land_count_ab, constants);
        } else {
            return get_prob_to_cast(constants.cmcs[card_index], required_a, required_b, land_count_a, land_count_b,
                                    land_count_ab, constants);
        }
    } else {
        unsigned char total_devotion = 0;
        float probability = 1.f;
        const unsigned char cmc = constants.cmcs[card_index];
        for (size_t i=0; i < num_requirements; i++) {
            const std::pair<std::array<unsigned char, NUM_COMBINATIONS>, unsigned char>& entry = color_requirement.first[i];
            const unsigned char required = entry.second;
            const std::array<unsigned char, NUM_COMBINATIONS>& color = entry.first;
            total_devotion += required;
            unsigned char land_count = 0;
            for (unsigned char j=1; j < NUM_COLORS + 1; j++) land_count += color[j] * lands[j].second;
            probability *= get_prob_to_cast(cmc, required, 0, land_count, 0, 0, constants);
        }
        unsigned char land_count = 0;
        for (size_t i=0; i < NUM_COMBINATIONS; i++) {
            const std::pair<Colors, unsigned char>& entry2 = lands[i];
            for (size_t j=0; j < num_requirements; j++) {
                const std::pair<std::array<unsigned char, NUM_COMBINATIONS>, unsigned char>& entry = color_requirement.first[j];
                if (entry.first[i] == 1) {
                    land_count += entry2.second;
                    break;
                }
            }
        }
        return probability * get_prob_to_cast(cmc, total_devotion, 0, land_count, 0, 0, constants);
    }
}

float calculate_synergy(const index_type card_index_1, const index_type card_index_2, const Variables& variables, const Constants& constants) {
    const float scaled = variables.similarity_multiplier * std::min(std::max(0.f, constants.similarities[card_index_1][card_index_2] - variables.similarity_clip),
                                                                    1 - variables.similarity_clip);
    if (card_index_1 == card_index_2) return variables.equal_cards_synergy;
    const float transformed = 1 / (1 - scaled) - 1;
    if (ISNAN(transformed)) return 0;
    else return std::min(transformed, MAX_SCORE);
}

float rating_oracle(const index_type card_index, const Lands&, const Variables& variables, const Pick& pick, const Constants& constants,
                    const float probability) {
#ifdef OPTIMIZE_RATINGS
    return probability * variables.ratings[pick.in_pack[card_index]];
#else
    return probability * constants.ratings[pick.in_pack[card_index]];
#endif
}

float pick_synergy_oracle(const index_type, const Lands&, const Variables&, const Pick&,
                          const Constants&, std::array<float, MAX_PICKED> probabilities, const index_type num_valid_indices,
                          const float probability, const std::array<float, MAX_PICKED>& synergies) {
    if (num_valid_indices == 0) return 0;
    float total_synergy = 0;
    for (size_t i=0; i < num_valid_indices; i++) total_synergy += probabilities[i] * synergies[i];
    return total_synergy * probability / (float)num_valid_indices;
}

float fixing_oracle(const index_type card_index, const Lands& lands, const Variables& variables, const Pick& pick, const Constants& constants) {
    const index_type card_real_index = pick.in_pack[card_index];
    if (constants.is_land[card_real_index]) {
        float overlap = 0;
        for (size_t i=0; i < NUM_COLORS; i++){
            if (constants.card_colors[card_real_index][i]) {
                unsigned char count = 0;
                for (size_t j=0; j < INCLUSION_MAP[i].size(); j++) count += lands[j].second;
                if (count >= LANDS_TO_INCLUDE_COLOR) overlap += MAX_SCORE / 5;
            }
        }
        if (constants.is_fetch[card_real_index]) return variables.is_fetch_multiplier * overlap;
        else if (constants.has_basic_land_types[card_real_index]) return variables.has_basic_types_multiplier * overlap;
        else return variables.is_regular_land_multiplier * overlap;
    } else return 0;
}

float internal_synergy_oracle(const index_type, const Lands&, const Variables&, const Pick&, const Constants&,
                              const std::array<float, MAX_PICKED>& probabilities, const index_type num_valid_indices,
                              const std::array<std::array<float, MAX_PICKED>, MAX_PICKED>& synergies) {
    if (num_valid_indices < 2) return 0;
    float internal_synergy = 0;
    for(index_type i=0; i < num_valid_indices; i++) {
        if (probabilities[i] > 0) {
            float card_synergy = 0;
            for (index_type j = 0; j < i; j++) card_synergy += probabilities[j] * synergies[i][j];
            internal_synergy += probabilities[i] * card_synergy;
        }
    }
    return 2 * internal_synergy / (float)(num_valid_indices * (num_valid_indices - 1));
}

template<size_t Size>
float sum_gated_rating(const Variables& variables, const std::array<index_type, Size>& indices,
                       const Constants& constants, const std::array<float, Size> probabilities,
                       const index_type num_valid_indices) {
    float result = 0;
    if (num_valid_indices == 0) return 0;
    for (index_type i=0; i < num_valid_indices; i++) {
#ifdef OPTIMIZE_RATINGS
        result += variables.ratings[indices[i]] * probabilities[i];
#else
        result += constants.ratings[indices[i]] * probabilities[i];
#endif
    }
    return result / (float) num_valid_indices;
}

float openness_oracle(const index_type, const Lands&, const Variables& variables, const Pick& pick, const Constants& constants,
                      const std::array<float, MAX_SEEN> probabilities, const index_type num_valid_indices) {
    return sum_gated_rating(variables, pick.seen, constants, probabilities, num_valid_indices);
}

float colors_oracle(const index_type, const Lands&, const Variables& variables, const Pick& pick, const Constants& constants,
                    const std::array<float, MAX_PICKED> probabilities, const index_type num_valid_indices) {
    return sum_gated_rating(variables, pick.picked, constants, probabilities, num_valid_indices);
}

float get_score(const index_type card_index, const Lands& lands, const Variables& variables, const Pick& pick,
                const float rating_weight, const float pick_synergy_weight, const float fixing_weight,
                const float internal_synergy_weight, const float openness_weight, const float colors_weight,
                const Constants& constants, const index_type num_valid_picked_indices, const index_type num_valid_seen_indices,
                const std::array<std::array<float, MAX_PICKED>, MAX_PICKED>& internal_synergies,
                const std::array<float, MAX_PICKED>& pick_synergies) {
    std::array<float, MAX_PICKED> picked_probabilities{0};
    std::array<float, MAX_SEEN> seen_probabilities{0};
    for (index_type i=0; i < num_valid_picked_indices; i++) {
        picked_probabilities[i] =
                std::max(get_casting_probability(lands, pick.picked[i], constants) - variables.prob_to_include, 0.f)
                * variables.prob_multiplier;
    }
    for (index_type i=0; i < num_valid_seen_indices; i++) {
        seen_probabilities[i] =
                std::max(get_casting_probability(lands, pick.seen[i], constants) - variables.prob_to_include, 0.f)
                * variables.prob_multiplier;
    }
    float card_casting_probability = get_casting_probability(lands, pick.in_pack[card_index], constants);
    const float rating_score = rating_oracle(card_index, lands, variables, pick, constants, card_casting_probability);
//    std::cout << rating_score << "*" << rating_weight;
    const float pick_synergy_score = pick_synergy_oracle(card_index, lands, variables, pick, constants, picked_probabilities,
                                                         num_valid_picked_indices, card_casting_probability, pick_synergies);
//    std::cout << " + " << pick_synergy_score << "*" << pick_synergy_weight;
    const float fixing_score = fixing_oracle(card_index, lands, variables, pick, constants);
//    std::cout << " + " << fixing_score << "*" << fixing_weight;
    const float internal_synergy_score = internal_synergy_oracle(card_index, lands, variables, pick, constants,
                                                                 picked_probabilities, num_valid_picked_indices, internal_synergies);
//    std::cout << " + " << internal_synergy_score << "*" << internal_synergy_weight;
    const float openness_score = openness_oracle(card_index, lands, variables, pick, constants, seen_probabilities,
                                                 num_valid_seen_indices);
//    std::cout << " + " << openness_score << "*" << openness_weight;
    const float colors_score = colors_oracle(card_index, lands, variables, pick, constants, picked_probabilities,
                                             num_valid_picked_indices);
//    std::cout << " + " << colors_score << "*" << colors_weight;
//    std::cout << std::endl;
    return rating_score*rating_weight + pick_synergy_score*pick_synergy_weight + fixing_score*fixing_weight
           + internal_synergy_score*internal_synergy_weight + openness_score*openness_weight + colors_score*colors_weight;
}

float do_climb(const index_type card_index, const Variables& variables, const Pick& pick, const Constants& constants,
               const index_type num_valid_picked_indices, const index_type num_valid_seen_indices,
               const std::array<std::array<float, MAX_PICKED>, MAX_PICKED>& internal_synergies) {
    float previous_score = -1;
    float current_score = 0;
    const float rating_weight = interpolate_weights(variables.rating_weights, pick);
    const float pick_synergy_weight = interpolate_weights(variables.pick_synergy_weights, pick);
    const float fixing_weight = interpolate_weights(variables.fixing_weights, pick);
    const float internal_synergy_weight = interpolate_weights(variables.internal_synergy_weights, pick);
    const float openness_weight = interpolate_weights(variables.openness_weights, pick);
    const float colors_weight = interpolate_weights(variables.colors_weights, pick);
    std::array<float, MAX_PICKED> pick_synergies{0};
    for (index_type i=0; i < num_valid_picked_indices; i++) {
        pick_synergies[i] = calculate_synergy(pick.picked[i], pick.in_pack[card_index], variables, constants);
    }
    Lands lands = DEFAULT_LANDS;
    std::array<bool, COLORS.size() + 1> previous_adds{false};
    std::array<bool, COLORS.size() + 1> previous_removes{false};
    while (previous_score < current_score) {
        previous_score = current_score;
        for(size_t remove_index=1; remove_index < COLORS.size() + 1; remove_index++) {
            if (lands[remove_index].second > 0 && !previous_adds[remove_index]) {
                bool breakout = false;
                for (size_t add_index=1; add_index < COLORS.size() + 1; add_index++) {
                    if (add_index == remove_index || previous_removes[add_index]) continue;
                    Lands new_lands = lands;
                    new_lands[remove_index].second -= 1;
                    new_lands[add_index].second += 1;
                    float score = get_score(card_index, new_lands, variables, pick, rating_weight, pick_synergy_weight,
                                            fixing_weight, internal_synergy_weight, openness_weight, colors_weight,
                                            constants, num_valid_picked_indices, num_valid_seen_indices,
                                            internal_synergies, pick_synergies);
                    if (score > current_score) {
//                        std::cout << "New score " << score << " remove_index " << remove_index << " add_index " << add_index << std::endl;
                        previous_adds[add_index] = true;
                        previous_removes[remove_index] = true;
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
    auto num_valid_indices = (index_type)pick.in_pack.size();
    for (size_t i=0; i < pick.in_pack.size(); i++) {
        if (pick.in_pack[i] == std::numeric_limits<index_type>::max()) {
            num_valid_indices = (index_type)i;
            break;
        }
    }
    auto num_valid_picked_indices = (index_type)pick.picked.size();
    auto num_valid_seen_indices = (index_type)pick.seen.size();
    for (size_t i=0; i < pick.picked.size(); i++) {
        if (pick.picked[i] == std::numeric_limits<index_type>::max()) {
            num_valid_picked_indices = (index_type)i;
            break;
        }
    }
    for (size_t i=0; i < pick.seen.size(); i++) {
        if (pick.seen[i] == std::numeric_limits<index_type>::max()) {
            num_valid_seen_indices = (index_type)i;
            break;
        }
    }
    std::array<std::array<float, MAX_PICKED>, MAX_PICKED> internal_synergies{{0}};
    for (index_type i=0; i < num_valid_picked_indices; i++) {
        for (index_type j=0; j < i; j++) {
            internal_synergies[i][j] = calculate_synergy(pick.picked[i], pick.picked[j], variables, constants);
        }
    }
    double max_score = 0;
    double denominator = 0;
    for (index_type i=0; i < num_valid_indices; i++) {
        const double score = do_climb(i, variables, pick, constants, num_valid_picked_indices, num_valid_seen_indices, internal_synergies);
//        std::cout << "Score " << i << ": " << score << std::endl;
        scores[i] = EXP(score / temperature);
        max_score = std::max(scores[i], max_score);
        denominator += scores[i];
    }
    bool best = true;
    for (index_type i=0; i < num_valid_indices; i++) best &= (pick.in_pack[i] == pick.chosen_card) == (scores[i] == max_score);
    for(index_type i=0; i < num_valid_indices; i++) {
        if (pick.in_pack[i] == pick.chosen_card) {
            if (scores[i] >= 0) {
//                std::cout << "Final Score: " << scores[i] << " / " << denominator << " = " << scores[i] / denominator << " -> " << -LOG(scores[i] / denominator) << std::endl;
                return {-LOG(scores[i] / denominator), best};
            } else {
                return {-1, false};
            }
        }
    }
    return {-2, false};
}

std::array<double, 4> get_batch_loss(const std::vector<Pick>& picks, const Variables& variables,
                                     const float temperature, const Constants& constants) {
    size_t count_correct = 0;
    std::array<double, 4> result{{0, 0, 0, 0}};
    for (size_t i=0; i < PICKS_PER_GENERATION; i++) {
        const std::pair<double, bool> pick_loss = calculate_loss(picks[i], variables, temperature, constants);
//        std::cout << "Finished pick " << i << " with Loss " << pick_loss.first << " and correctness " << pick_loss.second << std::endl;
        if (pick_loss.first < 0) return {pick_loss.first, -1, -1, -1};
        result[1] += pick_loss.first;
        if (pick_loss.second) count_correct++;
    }
    result[3] = count_correct / (double)PICKS_PER_GENERATION;
    result[2] = -LOG(result[3]);
    result[1] /= PICKS_PER_GENERATION;
    result[0] = CATEGORICAL_CROSSENTROPY_LOSS_WEIGHT * result[1]
                 + NEGATIVE_LOG_ACCURACY_LOSS_WEIGHT * result[2];
    return result;
}

#ifdef USE_SYCL
class calculate_batch_loss;

void run_simulations_on_device(const cl::sycl::device& device,
                               moodycamel::ConcurrentQueue<size_t>& concurrent_queue,
                               moodycamel::ProducerToken& producer_token,
                               const std::vector<Variables>& variables,
                               const std::vector<Pick>& picks,
                               const Constants& constants,
                               const float temperature,
                               std::vector<std::vector<std::pair<double, bool>>>& results) {
    cl::sycl::queue sycl_queue{device};
    const size_t num_groups = sycl_queue.get_device().get_info<cl::sycl::info::device::max_compute_units>();
    const size_t work_group_size = sycl_queue.get_device().get_info<cl::sycl::info::device::max_work_group_size>();
    const auto total_threads = num_groups * work_group_size;
    const size_t variables_to_process = (size_t)(FRACTION_OF_WORK_GROUPS * total_threads / 1024);
    std::vector<Variables> iteration_variables;
    iteration_variables.reserve(variables_to_process);
    std::vector<size_t> variable_indices;
    variable_indices.reserve(variables_to_process);
    cl::sycl::buffer<Pick, 1> picks_buffer(picks.data(), sycl::range<1>(picks.size()));
    cl::sycl::buffer<Constants, 1> constants_buffer(&constants, sycl::range<1>(1));
    cl::sycl::buffer<std::pair<double, bool>, 2> output_buffer{{variables_to_process, PICKS_PER_GENERATION}};
    while (true) {
        variable_indices.clear();
        iteration_variables.clear();
        for (size_t i=0; i < variables_to_process; i++) {
            size_t current_variables_index = 0;
            if(!concurrent_queue.try_dequeue_from_producer(producer_token, current_variables_index)) {
                break;
            }
            variable_indices.push_back(current_variables_index);
            iteration_variables.push_back(variables[current_variables_index]);
        }
        if (variable_indices.empty()) break;
        cl::sycl::buffer<Variables, 1> variables_buffer(iteration_variables.data(), sycl::range<1>(variables_to_process));
        try {
            /* Submit a command_group to execute from the queue. */
            sycl_queue.submit([&](cl::sycl::handler &cgh) {
                /* Create accessors for accessing the input and output data within the
                 * kernel. */
                auto variables_ptr = variables_buffer.get_access<cl::sycl::access::mode::read>(cgh);
                auto picks_ptr = picks_buffer.get_access<cl::sycl::access::mode::read>(cgh);
                auto constants_ptr = constants_buffer.get_access<cl::sycl::access::mode::read>(cgh);
                auto output_ptr = output_buffer.get_access<cl::sycl::access::mode::write>(cgh);

                cgh.parallel_for<class calculate_batch_loss>(sycl::range<2>{variable_indices.size(), PICKS_PER_GENERATION},
                                                             [=](cl::sycl::item<2> item_id) {
                                                                 const Variables& variables = variables_ptr[item_id[0]];
                                                                 const Constants &constants = constants_ptr[0];
                                                                 const Pick &pick = picks_ptr[item_id[1]];
                                                                 //for(size_t i=item_id[1]; i < PICKS_PER_GENERATION; i += item_id.get_range()[1]) {
                                                                     output_ptr[item_id] = calculate_loss(
                                                                             pick, variables, temperature, constants);
//                                                                 }
                                                             });
            });
            sycl_queue.wait();
            auto output_ptr = output_buffer.get_access<cl::sycl::access::mode::read>();
            for (size_t i=0; i < variable_indices.size(); i++) {
                for (size_t j=0; j < PICKS_PER_GENERATION; j++) {
                    results[variable_indices[i]][j] = output_ptr[i][j];
                }
            }
                    } catch (cl::sycl::exception& error) {
            std::cerr << error.what() << std::endl;
            std::rethrow_exception(std::current_exception());
        }
    }
}
#endif

std::array<std::array<double, 4>, POPULATION_SIZE> run_simulations(const std::vector<Variables>& variables,
                                                                   const std::vector<Pick>& picks, const float temperature,
                                                                   const std::shared_ptr<const Constants>& constants) {
    std::array<std::array<double, 4>, POPULATION_SIZE> results{{0,0,0,0}};
#ifdef USE_SYCL
    try {
        std::vector<std::vector<std::pair<double, bool>>> pick_results(POPULATION_SIZE, std::vector<std::pair<double, bool>>(PICKS_PER_GENERATION, {0, false}));
        moodycamel::ConcurrentQueue<size_t> concurrent_queue(POPULATION_SIZE);
        moodycamel::ProducerToken producer_token(concurrent_queue);
        std::vector<cl::sycl::queue> queues;
        for (size_t i=0; i < POPULATION_SIZE; i++) concurrent_queue.enqueue(producer_token, i);
        std::vector<std::thread> threads;
        auto plats = cl::sycl::platform::get_platforms();
        const auto& plat = plats.at(1);
        auto devices = plat.get_devices();
        threads.reserve(devices.size());
        for (const auto& device : devices) {
            threads.emplace_back([&]{ run_simulations_on_device(device, concurrent_queue, producer_token,
                                                                variables, picks, *constants, temperature, pick_results); });
        }
        if (threads.empty()) throw std::runtime_error{ "No OpenCL devices found." };
        for (auto& thread : threads) thread.join();
        for (size_t i = 0; i < POPULATION_SIZE; i++) {
            size_t count_correct = 0;
            for (size_t j = 0; j < PICKS_PER_GENERATION; j++) {
                // if (j % 1000 == 0 || pick_results[i][j].first <= 0 || ISINF(pick_results[i][j].first) || ISNAN(pick_results[i][j].first)) std::cout << i << "," << j << " " << pick_results[i][j].first << "," << pick_results[i][j].second << std::endl;
                results[i][1] += pick_results[i][j].first;
                if (pick_results[i][j].second) count_correct++;
            }
            results[i][3] = count_correct / (double) PICKS_PER_GENERATION;
            results[i][2] = -LOG(results[i][3]);
            results[i][1] /= PICKS_PER_GENERATION;
            results[i][0] = CATEGORICAL_CROSSENTROPY_LOSS_WEIGHT * results[i][1]
                             + NEGATIVE_LOG_ACCURACY_LOSS_WEIGHT * results[i][2];
        }
    } catch (cl::sycl::exception& e) {
        /* In the case of an exception being throw, print the error message and
         * rethrow it. */
        std::cerr << e.what() << std::endl;
        std::rethrow_exception(std::current_exception());
    }
#else
    /* std::array<std::future<std::array<double, 2>>, POPULATION_SIZE> futures; */
    /* std::array<std::thread, POPULATION_SIZE> threads; */
    /* for (size_t i=0; i < POPULATION_SIZE; i++) { */
    /*     std::packaged_task<std::array<double, 2>()> task( */
    /*             [&variables, i, &picks, temperature, &constants] { */
    /*                 return get_batch_loss(picks, variables[i], temperature, *constants); */
    /*             }); // wrap the function */
    /*     futures[i] = std::move(task.get_future());  // get a future */
    /*     std::thread t(std::move(task)); // launch on a thread */
    /*     threads[i] = std::move(t); */
    /* } */
    /* for (size_t i=0; i < POPULATION_SIZE; i++) { */
    /*     threads[i].join(); */
    /*     futures[i].wait(); */
    /*     results[i] = futures[i].get(); */
    /* } */
    for (size_t i=0; i < POPULATION_SIZE; i++) {
        results[i] = get_batch_loss(picks, variables[i], temperature, *constants);
    }
#endif
    return results;
}

Weights generate_weights(std::mt19937_64& gen) {
    Weights result{};
    std::uniform_real_distribution<float> weight_dist(MIN_WEIGHT, MAX_WEIGHT);
    for (size_t i=0; i < PACKS; i++) {
        for (size_t j=0; j < PACK_SIZE; j++) {
            result[i][j] = weight_dist(gen);
        }
    }
    return result;
}

int main(const int argc, const char* argv[]) {
    if (argc < 3) {
        return -1;
    }
    const std::shared_ptr<const Constants> constants = [](){ // NOLINT(cert-err58-cpp)
        std::shared_ptr<Constants> result_ptr = std::make_shared<Constants>();
        populate_constants("data/intToCard.json", *result_ptr);
        populate_prob_to_cast("data/probTable.json", *result_ptr);
        return result_ptr;
    }();
    std::random_device rd;
    size_t seed = rd();
    std::shared_ptr<const Variables> initial_variables;
    if (argc > 4) initial_variables = std::make_shared<Variables>(load_variables(argv[4]));
    if (argc > 3) seed = std::strtoull(argv[3], nullptr, 10);
    std::cout.precision(PRECISION);
    std::cout << std::fixed;
    const float temperature = std::strtof(argv[1], nullptr);
    const size_t generations = std::strtoull(argv[2], nullptr, 10);
    const std::vector<Pick> all_picks = load_picks("data/drafts/");
    const Variables best_variables = optimize_variables(temperature, all_picks, generations, constants, initial_variables, seed);
    save_variables(best_variables, "output/variables.json");
}
