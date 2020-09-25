#include "draftbot_optimization.h"

#define LOG(x) std::log(x)
#define SQRT(x) std::sqrt(x)
#define ISNAN(x) isnan(x)
#define ISINF(x) std::isinf(x)
#define EXP(x) std::exp(x)

__device__ constexpr unsigned char sum_with_mask(const unsigned char (&counts)[32], const unsigned char (&masks)[32]) {
    unsigned char acc = 0;
    for (size_t i=0; i < NUM_COMBINATIONS; i++) acc += counts[i] & masks[i];
    return acc;
}

__device__ std::tuple<unsigned char, unsigned char, unsigned char> sum_with_mask_2(const unsigned char (&counts)[32], const unsigned char (&masks_a)[32],
                                                                        const unsigned char (&masks_b)[32]) {
    std::tuple<unsigned char, unsigned char, unsigned char> accs{0, 0, 0};
    for (size_t i=0; i < NUM_COMBINATIONS; i++) {
        std::get<0>(accs) += masks_a[i] & ~masks_b[i] & counts[i];
        std::get<1>(accs) += ~masks_a[i] & masks_b[i] & counts[i];
        std::get<2>(accs) += masks_a[i] & masks_b[i] & counts[i];
    }
    return accs;
}

__device__ unsigned char sum_with_mask_3(const unsigned char (&counts)[32], const std::pair<unsigned char[NUM_COMBINATIONS], unsigned char> (&masks)[5]) {
    unsigned char acc = 0;
    for (size_t i=0; i < NUM_COMBINATIONS; i++) {
        unsigned char mask = masks[0].first[i];
        for (unsigned char j=1; j < 5; j++) mask |= masks[j].first[i];
        acc += mask & counts[i];
    }
    return acc;
}

__device__ unsigned char sum_with_mask_4(const unsigned char (&counts)[32], const std::pair<unsigned char[NUM_COMBINATIONS], unsigned char> (&masks)[5]) {
    unsigned char acc = 0;
    for (size_t i=0; i < NUM_COMBINATIONS; i++) {
        unsigned char mask = masks[0].first[i];
        for (unsigned char j=1; j < 5; j++) mask |= masks[j].first[i];
        acc += mask & counts[i];
    }
    return acc;
}

__device__ unsigned char sum_with_mask_5(const unsigned char (&counts)[32], const std::pair<unsigned char[NUM_COMBINATIONS], unsigned char> (&masks)[5]) {
    unsigned char acc = 0;
    for (size_t i=0; i < NUM_COMBINATIONS; i++) {
        unsigned char mask = masks[0].first[i];
        for (unsigned char j=1; j < 5; j++) mask |= masks[j].first[i];
        acc += mask & counts[i];
    }
    return acc;
}

__device__ constexpr float interpolate_weights(const Weights& weights, const Pick& pick) {
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

__device__ float get_casting_probability_0(const Lands& lands, const unsigned char cmc, const std::pair<unsigned char[NUM_COMBINATIONS], unsigned char> (&requirements)[5], const size_t offset, const Constants& constants) {
    return 1;
}

__device__ float get_casting_probability_1(const Lands& lands, const unsigned char cmc, const std::pair<unsigned char[NUM_COMBINATIONS], unsigned char> (&requirements)[5], const size_t offset, const Constants& constants) {
    return constants.get_prob_to_cast(offset, sum_with_mask(lands, requirements[0].first));
}

__device__ float get_casting_probability_2(const Lands& lands, const unsigned char cmc, const std::pair<unsigned char[NUM_COMBINATIONS], unsigned char> (&requirements)[5], const size_t offset, const Constants& constants) {
    const auto counts = sum_with_mask_2(lands, requirements[0].first, requirements[1].first);
    return constants.get_prob_to_cast(offset, std::get<0>(counts), std::get<1>(counts), std::get<2>(counts));
}

__device__ float get_casting_probability_3(const Lands& lands, const unsigned char cmc, const std::pair<unsigned char[NUM_COMBINATIONS], unsigned char> (&requirements)[5], const size_t offset, const Constants& constants) {
    unsigned char total_devotion = 0;
    float probability = 1.f;
    for (size_t i=0; i < 3; i++) {
        const auto& color = requirements[i].first;
        const auto& required = requirements[i].second;
        total_devotion += required;
        probability *= constants.get_prob_to_cast(cmc, required, sum_with_mask(lands, color));
    }
    unsigned char land_count = sum_with_mask_3(lands, requirements);
    return probability * constants.get_prob_to_cast(cmc, total_devotion, land_count);
}

__device__ float get_casting_probability_4(const Lands& lands, const unsigned char cmc, const std::pair<unsigned char[NUM_COMBINATIONS], unsigned char> (&requirements)[5], const size_t offset, const Constants& constants) {
    unsigned char total_devotion = 0;
    float probability = 1.f;
    for (size_t i=0; i < 4; i++) {
        const auto& color = requirements[i].first;
        const auto& required = requirements[i].second;
        total_devotion += required;
        probability *= constants.get_prob_to_cast(cmc, required, sum_with_mask(lands, color));
    }
    unsigned char land_count = sum_with_mask_4(lands, requirements);
    return probability * constants.get_prob_to_cast(cmc, total_devotion, land_count);
}

__device__ float get_casting_probability_5(const Lands& lands, const unsigned char cmc, const std::pair<unsigned char[NUM_COMBINATIONS], unsigned char> (&requirements)[5], const size_t offset, const Constants& constants) {
    unsigned char total_devotion = 0;
    float probability = 1.f;
    for (size_t i=0; i < 5; i++) {
        const auto& color = requirements[i].first;
        const auto& required = requirements[i].second;
        total_devotion += required;
        probability *= constants.get_prob_to_cast(cmc, required, sum_with_mask(lands, color));
    }
    unsigned char land_count = sum_with_mask_5(lands, requirements);
    return probability * constants.get_prob_to_cast(cmc, total_devotion, land_count);
}

__device__ float get_casting_probability(const Lands& lands, const unsigned short card_index, const Constants& constants) {
    const ColorRequirement& requirements = constants.color_requirements[card_index];
    switch(requirements.requirements_count) {
        case 0:
            return get_casting_probability_0(lands, constants.cmcs[card_index], requirements.requirements,  requirements.offset, constants);
        case 1:
            return get_casting_probability_1(lands, constants.cmcs[card_index],  requirements.requirements,  requirements.offset, constants);
        case 2:
            return get_casting_probability_2(lands, constants.cmcs[card_index],  requirements.requirements,  requirements.offset, constants);
        case 3:
            return get_casting_probability_3(lands, constants.cmcs[card_index],  requirements.requirements,  requirements.offset, constants);
        case 4:
            return get_casting_probability_4(lands, constants.cmcs[card_index],  requirements.requirements,  requirements.offset, constants);
        case 5:
            return get_casting_probability_5(lands, constants.cmcs[card_index],  requirements.requirements,  requirements.offset, constants);
    }
    return 0;
}

__device__ constexpr float calculate_synergy(const index_type card_index_1, const index_type card_index_2, const Variables& variables, const Constants& constants) {
    const float scaled = variables.similarity_multiplier * std::min(std::max(0.f, constants.similarities[card_index_1][card_index_2] - variables.similarity_clip),
                                                                    1 - variables.similarity_clip);
    if (card_index_1 == card_index_2) return variables.equal_cards_synergy;
    const float transformed = 1 / (1 - scaled) - 1;
    if (ISNAN(transformed)) return 0;
    else return std::min(transformed, 10.f);
}

__device__ constexpr float rating_oracle(const index_type card_index, const Lands&, const Variables& variables, const Pick& pick, const Constants& constants,
                              const float probability) {
#ifdef OPTIMIZE_RATINGS
    return probability * variables.ratings[pick.in_pack[card_index]];
#else
    return probability * constants.ratings[pick.in_pack[card_index]];
#endif
}

__device__ constexpr float pick_synergy_oracle(const index_type, const Lands&, const Variables&, const Pick&,
                                    const Constants&, const float (&probabilities)[MAX_PICKED], const index_type num_valid_indices,
                                    const float probability, const float (&synergies)[MAX_PICKED]) {
    if (num_valid_indices == 0) return 0;
    float total_synergy = 0;
    for (size_t i=0; i < num_valid_indices; i++) total_synergy += probabilities[i] * synergies[i];
    return total_synergy * probability / (float)num_valid_indices;
}

__device__ constexpr float fixing_oracle(const index_type card_index, const Lands& lands, const Variables& variables, const Pick& pick, const Constants& constants) {
    constexpr size_t INCLUSION_MAP[NUM_COLORS][16]{
           {1, 6,10, 11, 14, 16, 17, 20, 21, 23, 24, 27, 28, 29, 30, 31},
           {2, 6, 7, 12, 15, 16, 17, 18, 22, 24, 25, 26, 28, 29, 30, 31},
           {3, 7, 8, 11, 13, 17, 18, 19, 21, 23, 25, 26, 27, 29, 30, 31},
           {4, 8, 9, 12, 14, 18, 19, 20, 21, 22, 24, 26, 27, 28, 30, 31},
           {5, 9,10, 13, 15, 16, 19, 20, 22, 23, 25, 26, 27, 28, 29, 31},
   };
    const index_type card_real_index = pick.in_pack[card_index];
    if (constants.is_land[card_real_index]) {
        float overlap = 0;
        for (size_t i=0; i < NUM_COLORS; i++){
            if (constants.card_colors[card_real_index][i]) {
                unsigned char count = 0;
                for (size_t j=0; j < 16; j++) count += lands[INCLUSION_MAP[i][j]];
                if (count >= LANDS_TO_INCLUDE_COLOR) overlap += MAX_SCORE / 5;
            }
        }
        if (constants.is_fetch[card_real_index]) return variables.is_fetch_multiplier * overlap;
        else if (constants.has_basic_land_types[card_real_index]) return variables.has_basic_types_multiplier * overlap;
        else return variables.is_regular_land_multiplier * overlap;
    } else return 0;
}

__device__ constexpr float internal_synergy_oracle(const index_type, const Lands&, const Variables&, const Pick&, const Constants&,
                                        const float (&probabilities)[MAX_PICKED], const index_type num_valid_indices,
                                        const float (&synergies)[MAX_PICKED][MAX_PICKED]) {
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
__device__ constexpr float sum_gated_rating(const Variables& variables, const index_type (&indices)[Size],
                                 const Constants& constants, const float (&probabilities)[Size],
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

__device__ float openness_oracle(const index_type, const Lands&, const Variables& variables, const Pick& pick, const Constants& constants,
                      const float (&probabilities)[MAX_SEEN], const index_type num_valid_indices) {
    return sum_gated_rating(variables, pick.seen, constants, probabilities, num_valid_indices);
}

__device__ constexpr float colors_oracle(const index_type, const Lands&, const Variables& variables, const Pick& pick, const Constants& constants,
                              const float (&probabilities)[MAX_PICKED], const index_type num_valid_indices) {
    return sum_gated_rating(variables, pick.picked, constants, probabilities, num_valid_indices);
}

__device__ float get_score(const index_type card_index, const Lands& lands, const Variables& variables, const Pick& pick,
                const float rating_weight, const float pick_synergy_weight, const float fixing_weight,
                const float internal_synergy_weight, const float openness_weight, const float colors_weight,
                const Constants& constants, const index_type num_valid_picked_indices, const index_type num_valid_seen_indices,
                const float (&internal_synergies)[MAX_PICKED][MAX_PICKED],
                const float (&pick_synergies)[MAX_PICKED]) {
    float picked_probabilities[MAX_PICKED];
    float seen_probabilities[MAX_SEEN];
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

__device__ float do_climb(const index_type card_index, const Variables& variables, const Pick& pick, const Constants& constants,
               const index_type num_valid_picked_indices, const index_type num_valid_seen_indices,
               const float (&internal_synergies)[MAX_PICKED][MAX_PICKED]) {
    constexpr Lands DEFAULT_LANDS{0, 4, 4, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    float previous_score = -1;
    float current_score = 0;
    const float rating_weight = interpolate_weights(variables.rating_weights, pick);
    const float pick_synergy_weight = interpolate_weights(variables.pick_synergy_weights, pick);
    const float fixing_weight = interpolate_weights(variables.fixing_weights, pick);
    const float internal_synergy_weight = interpolate_weights(variables.internal_synergy_weights, pick);
    const float openness_weight = interpolate_weights(variables.openness_weights, pick);
    const float colors_weight = interpolate_weights(variables.colors_weights, pick);
    float pick_synergies[MAX_PICKED];
    for (index_type i=0; i < num_valid_picked_indices; i++) {
        pick_synergies[i] = calculate_synergy(pick.picked[i], pick.in_pack[card_index], variables, constants);
    }
    Lands lands;
    for (size_t i=0; i < NUM_COMBINATIONS; i++) lands[i] = DEFAULT_LANDS[i];
    while (previous_score < current_score) {
        previous_score = current_score;
        /* Lands next_lands = lands; */
        /* size_t added_index=0; */
        /* size_t removed_index=0; */
        for(size_t remove_index=1; remove_index < COLORS.size() + 1; remove_index++) {
            if (lands[remove_index] > 0) {
                bool breakout = false;
                for (size_t add_index=1; add_index < COLORS.size() + 1; add_index++) {
                    if (add_index != remove_index) {
                        Lands new_lands;
                        for (size_t i=0; i < NUM_COMBINATIONS; i++) new_lands[i] = lands[i];
                        new_lands[remove_index] -= 1;
                        new_lands[add_index] += 1;
                        float score = get_score(card_index, new_lands, variables, pick, rating_weight, pick_synergy_weight,
                                                fixing_weight, internal_synergy_weight, openness_weight, colors_weight,
                                                constants, num_valid_picked_indices, num_valid_seen_indices,
                                                internal_synergies, pick_synergies);
                        if (score > current_score) {
                            current_score = score;
                            for (size_t i=0; i < NUM_COMBINATIONS; i++) lands[i] = new_lands[i];
                            breakout = true;
                            break;
                        }
                    }
                }
                if (breakout) break;
            }
        }
    }
    return current_score;
}

__device__ std::pair<double, bool> calculate_loss(const Pick& pick, const Variables& variables, const float temperature, const Constants& constants) {
    double scores[MAX_PACK_SIZE]{0};
    auto num_valid_indices = (index_type)MAX_PACK_SIZE;
    for (size_t i=0; i < MAX_PACK_SIZE; i++) {
        if (pick.in_pack[i] == std::numeric_limits<index_type>::max()) {
            num_valid_indices = (index_type)i;
            break;
        }
    }
    auto num_valid_picked_indices = (index_type)MAX_PICKED;
    auto num_valid_seen_indices = (index_type)MAX_SEEN;
    for (size_t i=0; i < MAX_PICKED; i++) {
        if (pick.picked[i] == std::numeric_limits<index_type>::max()) {
            num_valid_picked_indices = (index_type)i;
            break;
        }
    }
    for (size_t i=0; i < MAX_SEEN; i++) {
        if (pick.seen[i] == std::numeric_limits<index_type>::max()) {
            num_valid_seen_indices = (index_type)i;
            break;
        }
    }
    float internal_synergies[MAX_PICKED][MAX_PICKED]{{0}};
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

__global__ void run_thread(const Constants* constants_ptr, const Variables* variables_ptr, const Pick* picks_ptr, const float temperature,
                           double* loss_results_ptr, bool* accuracy_results_ptr, const size_t variables_offset) {
    size_t variable_id = variables_offset + blockIdx.x;
    size_t initial_pick_id = blockIdx.y * gridDim.x + threadIdx.x;
    size_t pick_stride = gridDim.y * blockDim.x;
    for (size_t i=initial_pick_id; i < PICKS_PER_GENERATION; i += pick_stride) {
        std::pair<double, bool> result = calculate_loss(picks_ptr[i], variables_ptr[variable_id], temperature, *constants_ptr);
        loss_results_ptr[variable_id * PICKS_PER_GENERATION + i] = result.first;
        accuracy_results_ptr[variable_id * PICKS_PER_GENERATION + i] = result.second;
    }
}

std::array<std::array<double, 4>, POPULATION_SIZE> run_simulations_cuda(const std::vector<Variables>& variables,
                                                                        const std::vector<Pick>& picks, const float temperature,
                                                                        const std::shared_ptr<const Constants>& constants) {
    Constants* constants_ptr;
    Variables* variables_ptr;
    Pick* picks_ptr;
    double *loss_results_ptr;
    bool* accuracy_results_ptr;
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);
    std::cout << "cooperativeMultiDeviceLaunch: " << props.cooperativeMultiDeviceLaunch << " maxBlocksPerMultiProcessor: "
              << props.maxBlocksPerMultiProcessor << " maxThreadsPerMultiProcessor: " << props.maxThreadsPerMultiProcessor
              << " maxThreadsDim[0]: " << props.maxThreadsDim[0] << " maxThreadsDim[1]: " << props.maxThreadsDim[1]
              << " maxThreadsDim[2]: " << props.maxThreadsDim[2] << " maxThreadsPerBlock: " << props.maxThreadsPerBlock
              << " multiProcessorCount: " << props.multiProcessorCount << " warpSize: " << props.warpSize << std::endl;
    cudaError_t cuda_error = cudaMallocManaged(&constants_ptr, sizeof(Constants));
    std::cout << cudaGetErrorString(cuda_error) << std::endl;
    cuda_error = cudaMemcpy(constants_ptr, constants.get(), sizeof(Constants), cudaMemcpyHostToHost);
    std::cout << cudaGetErrorString(cuda_error) << std::endl;
    cuda_error = cudaMallocManaged(&variables_ptr, sizeof(Variables) * variables.size());
    std::cout << cudaGetErrorString(cuda_error) << std::endl;
    cuda_error = cudaMemcpy(variables_ptr, variables.data(), sizeof(Variables) * variables.size(), cudaMemcpyHostToHost);
    std::cout << cudaGetErrorString(cuda_error) << std::endl;
    cuda_error = cudaMallocManaged(&picks_ptr, sizeof(Pick) * picks.size());
    std::cout << cudaGetErrorString(cuda_error) << std::endl;
    cuda_error = cudaMemcpy(picks_ptr, picks.data(), sizeof(Pick) * picks.size(), cudaMemcpyHostToHost);
    std::cout << cudaGetErrorString(cuda_error) << std::endl;
    cuda_error = cudaMallocManaged(&loss_results_ptr, sizeof(double) * variables.size() * picks.size());
    std::cout << cudaGetErrorString(cuda_error) << std::endl;
    cuda_error = cudaMallocManaged(&accuracy_results_ptr, sizeof(bool) * variables.size() * picks.size());
    std::cout << cudaGetErrorString(cuda_error) << std::endl;
    const dim3 gridDim(34, 2 * 16);
    const dim3 blockDim(1024 / 16);
    run_thread<<<gridDim, blockDim>>>(constants_ptr, variables_ptr, picks_ptr, temperature,
                                      loss_results_ptr, accuracy_results_ptr, 0);
    cudaDeviceSynchronize();
    std::array<std::array<double, 4>, POPULATION_SIZE> results;
    for (size_t i = 0; i < POPULATION_SIZE; i++) {
        size_t count_correct = 0;
        for (size_t j = 0; j < PICKS_PER_GENERATION; j++) {
            // if (j % 1000 == 0 || pick_results[i][j].first <= 0 || ISINF(pick_results[i][j].first) || ISNAN(pick_results[i][j].first)) std::cout << i << "," << j << " " << pick_results[i][j].first << "," << pick_results[i][j].second << std::endl;
            results[i][1] += loss_results_ptr[i * picks.size() + j];
            if (accuracy_results_ptr[i * picks.size() + j]) count_correct++;
        }
        results[i][3] = count_correct / (double) PICKS_PER_GENERATION;
        results[i][2] = -LOG(results[i][3]);
        results[i][1] /= PICKS_PER_GENERATION;
        results[i][0] = CATEGORICAL_CROSSENTROPY_LOSS_WEIGHT * results[i][1]
                        + NEGATIVE_LOG_ACCURACY_LOSS_WEIGHT * results[i][2];
    }
    cudaFree(constants_ptr);
    cudaFree(variables_ptr);
    cudaFree(picks_ptr);
    cudaFree(loss_results_ptr);
    cudaFree(accuracy_results_ptr);
    return results;
}