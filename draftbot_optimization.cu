#include "cuda_fp16.h"

#include "draftbot_optimization.h"

#define LOG(x) std::log(x)
#define SQRT(x) std::sqrt(x)
#define ISNAN(x) isnan(x)
#define ISINF(x) std::isinf(x)
#define EXP(x) std::exp(x)

struct HalfExpandedPick {
    index_type in_pack[MAX_PACK_SIZE]{std::numeric_limits<index_type>::max()}; // 2 * MAX_PACK_SIZE
    index_type in_pack_count{0};                                               // 2
    index_type seen[MAX_SEEN]{std::numeric_limits<index_type>::max()};         // 2 * MAX_SEEN
    index_type seen_count{0};                                                  // 2
    index_type picked[MAX_PICKED]{std::numeric_limits<index_type>::max()};     // 2 * MAX_PICKED
    index_type picked_count{0};                                                // 2
    unsigned char pack_num{0};                                                 // 1
    unsigned char pick_num{0};                                                 // 1
    unsigned char pack_size{0};                                                // 1
    unsigned char packs{0};                                                    // 1
    index_type chosen_card{0};                                                 // 2
    __half in_pack_similarities[MAX_PACK_SIZE][MAX_PICKED];                    // 2 * MAX_PACK_SIZE * MAX_PICKED
    ColorRequirement in_pack_color_requirements[MAX_PACK_SIZE];                // 56 * MAX_PACK_SIZE
    unsigned char in_pack_cmcs[MAX_PACK_SIZE];                                 // MAX_PACK_SIZE
    bool in_pack_is_land[MAX_PACK_SIZE];                                       // MAX_PACK_SIZE
    bool in_pack_card_colors[MAX_PACK_SIZE][5];                                // 5 * MAX_PACK_SIZE
    bool in_pack_is_fetch[MAX_PACK_SIZE];                                      // MAX_PACK_SIZE
    bool in_pack_has_basic_land_types[MAX_PACK_SIZE];                          // MAX_PACK_SIZE
    ColorRequirement seen_color_requirements[MAX_SEEN];                        // 56 * MAX_SEEN
    unsigned char seen_cmcs[MAX_SEEN];                                         // MAX_SEEN
    ColorRequirement picked_color_requirements[MAX_PICKED];                    // 56 * MAX_PICKED
    unsigned char picked_cmcs[MAX_PICKED];                                     // MAX_PICKED
    __half picked_similarities[MAX_PICKED][MAX_PICKED];                        // 2 * MAX_PICKED * MAX_PICKED
    // MAX_PACK_SIZE * (67 + 2 * MAX_PICKED) + MAX_SEEN * 59 + MAX_PICKED * (2 * MAX_PICKED + 59) + 12

    HalfExpandedPick(const ExpandedPick& pick) {
        in_pack_count = pick.in_pack_count;
        seen_count = pick.seen_count;
        picked_count = pick.picked_count;
        for (size_t i=0; i < in_pack_count; i++) {
            in_pack[i] = pick.in_pack[i];
            in_pack_cmcs[i] = pick.in_pack_cmcs[i];
            in_pack_is_land[i] = pick.in_pack_is_land[i];
            for (size_t j=0; j < 5; j++) in_pack_card_colors[i][j] = pick.in_pack_card_colors[i][j];
            in_pack_is_fetch[i] = pick.in_pack_is_fetch[i];
            in_pack_has_basic_land_types[i] = pick.in_pack_has_basic_land_types[i];
            in_pack_color_requirements[i] = pick.in_pack_color_requirements[i];
            for (size_t j=0; j < picked_count; j++) in_pack_similarities[i][j] = __float2half(pick.in_pack_similarities[i][j]);
        }
        for (size_t i=0; i < seen_count; i++) {
           seen[i] = pick.seen[i];
           seen_color_requirements[i] = pick.seen_color_requirements[i];
           seen_cmcs[i] = pick.seen_cmcs[i];
        }
        for (size_t i=0; i < picked_count; i++) {
            picked[i] = pick.picked[i];
            picked_color_requirements[i] = pick.picked_color_requirements[i];
            picked_cmcs[i] = pick.picked_cmcs[i];
            for (size_t j=0; j < i; j++) {
                picked_similarities[i][j] = __float2half(pick.picked_similarities[i][j]);
            }
        }
        pack_num = pick.pack_num;
        pick_num = pick.pick_num;
        pack_size = pick.pack_size;
        packs = pick.packs;
        chosen_card = pick.chosen_card;
   }
};

using HalfWeights = __half[PACKS][PACK_SIZE];

struct HalfVariables {
    HalfWeights rating_weights;
    HalfWeights colors_weights;
    HalfWeights fixing_weights;
    HalfWeights internal_synergy_weights;
    HalfWeights pick_synergy_weights;
    HalfWeights openness_weights;
#ifdef OPTIMIZE_RATINGS
    __half ratings[NUM_CARDS];
#endif
    __half prob_to_include;
    __half prob_multiplier;
    __half similarity_clip;
    __half similarity_multiplier;
    __half is_fetch_multiplier;
    __half has_basic_types_multiplier;
    __half is_regular_land_multiplier;
    __half equal_cards_synergy;

    HalfVariables() = default;
    HalfVariables& operator=(const HalfVariables& other) = default;

    HalfVariables(const Variables& variables) {
       for (size_t i=0; i < PACKS; i++) for (size_t j=0; j < PACK_SIZE; j++) rating_weights[i][j] = __float2half(variables.rating_weights[i][j]);
       for (size_t i=0; i < PACKS; i++) for (size_t j=0; j < PACK_SIZE; j++) pick_synergy_weights[i][j] = __float2half(variables.pick_synergy_weights[i][j]);
       for (size_t i=0; i < PACKS; i++) for (size_t j=0; j < PACK_SIZE; j++) fixing_weights[i][j] = __float2half(variables.fixing_weights[i][j]);
       for (size_t i=0; i < PACKS; i++) for (size_t j=0; j < PACK_SIZE; j++) internal_synergy_weights[i][j] = __float2half(variables.internal_synergy_weights[i][j]);
       for (size_t i=0; i < PACKS; i++) for (size_t j=0; j < PACK_SIZE; j++) openness_weights[i][j] = __float2half(variables.openness_weights[i][j]);
       for (size_t i=0; i < PACKS; i++) for (size_t j=0; j < PACK_SIZE; j++) colors_weights[i][j] = __float2half(variables.colors_weights[i][j]);
#ifdef OPTIMIZE_RATINGS
       for (size_t i=0; i < NUM_CARDS; i++) ratings[i] = __float2half(variables.ratings[i]);
#endif
       prob_to_include = __float2half(variables.prob_to_include);
       prob_multiplier = __float2half(variables.prob_multiplier);
       similarity_clip = __float2half(variables.similarity_clip);
       similarity_multiplier = __float2half(variables.similarity_multiplier);
       is_fetch_multiplier = __float2half(variables.is_fetch_multiplier);
       has_basic_types_multiplier = __float2half(variables.has_basic_types_multiplier);
       is_regular_land_multiplier = __float2half(variables.is_regular_land_multiplier);
       equal_cards_synergy = __float2half(variables.equal_cards_synergy);
    }
};

#ifdef CONSIDER_NON_BASICS
__device__ unsigned char sum_with_mask(const unsigned char (&counts)[32], const unsigned char (&masks)[32]) {
#else
__device__ unsigned char sum_with_mask(const unsigned char (&counts)[32], const unsigned char (&masks)[8]) {
#endif
    unsigned char acc = 0;
    size_t big_acc = 0;
#ifdef CONSIDER_NON_BASICS
    for (size_t i=0; i < 32 / sizeof(size_t); i++) {
        big_acc += reinterpret_cast<const size_t*>(masks)[i] & reinterpret_cast<const size_t*>(counts)[i];
    }
    for (size_t j=0; j < sizeof(size_t); j++) {
#else
    big_acc = reinterpret_cast<const size_t*>(masks)[0] & reinterpret_cast<const size_t*>(counts)[0];
    for (size_t j=1; j < 1 + COLORS.size(); j++) {
#endif
        acc += reinterpret_cast<const unsigned char*>(&big_acc)[j];
    }
    return acc;
}

#ifdef CONSIDER_NON_BASICS
__device__ std::tuple<unsigned char, unsigned char, unsigned char> sum_with_mask_2(const unsigned char (&counts)[32], const unsigned char (&masks_a)[32],
                                                                        const unsigned char (&masks_b)[32]) {
#else
__device__ std::tuple<unsigned char, unsigned char, unsigned char> sum_with_mask_2(const unsigned char (&counts)[32], const unsigned char (&masks_a)[8],
                                                                        const unsigned char (&masks_b)[8]) {
#endif
    std::tuple<unsigned char, unsigned char, unsigned char> accs{0, 0, 0};
    size_t big_acc_a = 0;
    size_t big_acc_b = 0;
    size_t big_acc_ab = 0;
#ifdef CONSIDER_NON_BASICS
    for (size_t i=0; i < 32 / sizeof(size_t); i++) {
        const size_t value = reinterpret_cast<const size_t*>(counts)[i];
        const size_t mask_a = reinterpret_cast<const size_t*>(masks_a)[i];
        const size_t mask_b = reinterpret_cast<const size_t*>(masks_b)[i];
        const size_t masked_a = value & mask_a;
        big_acc_a += ~mask_b & masked_a;
        big_acc_b += ~mask_a & mask_b & value;
        big_acc_ab += mask_b & masked_a;
    }
    for (size_t j=0; j < sizeof(size_t); j++) {
#else
    const size_t value = reinterpret_cast<const size_t*>(counts)[0];
    const size_t mask_a = reinterpret_cast<const size_t*>(masks_a)[0];
    const size_t mask_b = reinterpret_cast<const size_t*>(masks_b)[0];
    const size_t masked_a = value & mask_a;
    big_acc_a = ~mask_b & masked_a;
    big_acc_b = ~mask_a & mask_b & value;
    big_acc_ab = mask_b & masked_a;
    for (size_t j=1; j < 1 + COLORS.size(); j++) {
#endif
        std::get<0>(accs) += reinterpret_cast<const unsigned char*>(&big_acc_a)[j];
        std::get<1>(accs) += reinterpret_cast<const unsigned char*>(&big_acc_b)[j];
        std::get<2>(accs) += reinterpret_cast<const unsigned char*>(&big_acc_ab)[j];
    }
    return accs;
}

#ifdef CONSIDER_NON_BASICS
__device__ unsigned char sum_with_mask_3(const unsigned char (&counts)[32], const std::pair<unsigned char[NUM_COMBINATIONS], unsigned char> (&masks)[5]) {
#else
__device__ unsigned char sum_with_mask_3(const unsigned char (&counts)[32], const std::pair<unsigned char[8], unsigned char> (&masks)[5]) {
#endif
    unsigned char acc = 0;
    size_t big_acc = 0;
#ifdef CONSIDER_NON_BASICS
    for (size_t i=0; i < 32 / sizeof(size_t); i++) {
        big_acc += reinterpret_cast<const size_t*>(counts)[i] & reinterpret_cast<const size_t*>(masks[0].first)[i]
                 & reinterpret_cast<const size_t*>(masks[1].first)[i] & reinterpret_cast<const size_t*>(masks[2].first)[i];
    }
    for (size_t j=0; j < sizeof(size_t); j++) {
#else
    big_acc = reinterpret_cast<const size_t*>(counts)[0] & reinterpret_cast<const size_t*>(masks[0].first)[0]
            & reinterpret_cast<const size_t*>(masks[1].first)[0] & reinterpret_cast<const size_t*>(masks[2].first)[0];
    for (size_t j=1; j < 1 + COLORS.size(); j++) {
#endif
        acc += reinterpret_cast<const unsigned char*>(&big_acc)[j];
    }
    return acc;
}

#ifdef CONSIDER_NON_BASICS
__device__ unsigned char sum_with_mask_4(const unsigned char (&counts)[32], const std::pair<unsigned char[NUM_COMBINATIONS], unsigned char> (&masks)[5]) {
#else
__device__ unsigned char sum_with_mask_4(const unsigned char (&counts)[32], const std::pair<unsigned char[8], unsigned char> (&masks)[5]) {
#endif
    unsigned char acc = 0;
    size_t big_acc = 0;
#ifdef CONSIDER_NON_BASICS
    for (size_t i=0; i < 32 / sizeof(size_t); i++) {
        big_acc += reinterpret_cast<const size_t*>(counts)[i] & reinterpret_cast<const size_t*>(masks[0].first)[i]
                 & reinterpret_cast<const size_t*>(masks[1].first)[i] & reinterpret_cast<const size_t*>(masks[2].first)[i]
                 & reinterpret_cast<const size_t*>(masks[3].first)[i];
    }
    for (size_t j=0; j < sizeof(size_t); j++) {
#else
    big_acc = reinterpret_cast<const size_t*>(counts)[0] & reinterpret_cast<const size_t*>(masks[0].first)[0]
            & reinterpret_cast<const size_t*>(masks[1].first)[0] & reinterpret_cast<const size_t*>(masks[2].first)[0]
            & reinterpret_cast<const size_t*>(masks[3].first)[0];
    for (size_t j=1; j < 1 + COLORS.size(); j++) {
#endif
        acc += reinterpret_cast<const unsigned char*>(&big_acc)[j];
    }
    return acc;
}

#ifdef CONSIDER_NON_BASICS
__device__ unsigned char sum_with_mask_5(const unsigned char (&counts)[32], const std::pair<unsigned char[NUM_COMBINATIONS], unsigned char> (&masks)[5]) {
#else
__device__ unsigned char sum_with_mask_5(const unsigned char (&counts)[32], const std::pair<unsigned char[8], unsigned char> (&masks)[5]) {
#endif
    unsigned char acc = 0;
    size_t big_acc = 0;
#ifdef CONSIDER_NON_BASICS
    for (size_t i=0; i < 32 / sizeof(size_t); i++) {
        big_acc += reinterpret_cast<const size_t*>(counts)[i] & reinterpret_cast<const size_t*>(masks[0].first)[i]
                 & reinterpret_cast<const size_t*>(masks[1].first)[i] & reinterpret_cast<const size_t*>(masks[2].first)[i]
                 & reinterpret_cast<const size_t*>(masks[3].first)[i] & reinterpret_cast<const size_t*>(masks[4].first)[i];
    }
    for (size_t j=0; j < sizeof(size_t); j++) {
#else
    big_acc = reinterpret_cast<const size_t*>(counts)[0] & reinterpret_cast<const size_t*>(masks[0].first)[0]
            & reinterpret_cast<const size_t*>(masks[1].first)[0] & reinterpret_cast<const size_t*>(masks[2].first)[0]
            & reinterpret_cast<const size_t*>(masks[3].first)[0] & reinterpret_cast<const size_t*>(masks[4].first)[0];
    for (size_t j=1; j < 1 + COLORS.size(); j++) {
#endif
        acc += reinterpret_cast<const unsigned char*>(&big_acc)[j];
    }
    return acc;
}

__device__ __half interpolate_weights(const HalfWeights& weights, const HalfExpandedPick& pick) {
    const __half x_index = __ull2half_rd(PACKS) * __int2half_rd(pick.pack_num) / __int2half_rd(pick.packs);
    const __half y_index = __ull2half_rd(PACK_SIZE) * __int2half_rd(pick.pick_num) / __int2half_rd(pick.pack_size);
    const auto floor_x_index = __half2int_rd(x_index);
    const auto floor_y_index = __half2int_rd(y_index);
    const auto ceil_x_index = min((size_t)floor_x_index + 1, PACKS - 1);
    const auto ceil_y_index = min((size_t)floor_y_index + 1, PACK_SIZE - 1);
    const __half x_index_mod_one = x_index - __int2half_rd(floor_x_index);
    const __half y_index_mode_one = y_index - __int2half_rd(floor_y_index);
    const __half inv_x_index_mod_one = __int2half_rd(1) - x_index_mod_one;
    const __half inv_y_index_mod_one = __int2half_rd(1) - y_index_mode_one;
    const __half XY = x_index_mod_one * y_index_mode_one;
    const __half Xy = x_index_mod_one * inv_y_index_mod_one;
    const __half xY = inv_x_index_mod_one * y_index_mode_one;
    const __half xy = inv_x_index_mod_one * inv_y_index_mod_one;
    const __half XY_weight = weights[ceil_x_index][ceil_y_index];
    const __half Xy_weight = weights[ceil_x_index][floor_y_index];
    const __half xY_weight = weights[floor_x_index][ceil_y_index];
    const __half xy_weight = weights[floor_x_index][floor_y_index];
    return XY * XY_weight + Xy * Xy_weight + xY * xY_weight + xy * xy_weight;
}

__device__ __half get_prob_to_cast(const __half *prob_to_cast, const size_t cmc, const size_t required_a, const size_t land_count_a) noexcept {
    return prob_to_cast[(((cmc << PROB_DIM_1_EXP) | required_a) << (PROB_DIM_2_EXP + PROB_DIM_3_EXP + PROB_DIM_4_EXP + PROB_DIM_5_EXP)) | land_count_a];
}

__device__ __half get_prob_to_cast(const __half *prob_to_cast, const size_t offset, const size_t land_count_a) noexcept {
    return prob_to_cast[offset | land_count_a];
}

__device__ __half get_prob_to_cast(const __half *prob_to_cast, const size_t offset, const size_t land_count_a, const size_t land_count_b, const size_t land_count_ab) noexcept {
    return prob_to_cast[offset | (((land_count_ab << PROB_DIM_4_EXP) | land_count_b) << PROB_DIM_5_EXP) | land_count_a];
}

#ifdef CONSIDER_NON_BASICS
__device__ __half get_casting_probability_0(const Lands&, const unsigned char, const std::pair<unsigned char[NUM_COMBINATIONS], unsigned char> (&)[5], const size_t, const __half*) {
#else
__device__ __half get_casting_probability_0(const Lands&, const unsigned char, const std::pair<unsigned char[8], unsigned char> (&)[5], const size_t, const __half*) {
#endif
    return 1;
}

#ifdef CONSIDER_NON_BASICS
__device__ __half get_casting_probability_1(const Lands& lands, const unsigned char cmc, const std::pair<unsigned char[NUM_COMBINATIONS], unsigned char> (&requirements)[5], const size_t offset, const __half *prob_to_cast) {
#else
__device__ __half get_casting_probability_1(const Lands& lands, const unsigned char cmc, const std::pair<unsigned char[8], unsigned char> (&requirements)[5], const size_t offset, const __half *prob_to_cast) {
#endif
    return get_prob_to_cast(prob_to_cast, offset, sum_with_mask(lands, requirements[0].first));
}

#ifdef CONSIDER_NON_BASICS
__device__ __half get_casting_probability_2(const Lands& lands, const unsigned char cmc, const std::pair<unsigned char[NUM_COMBINATIONS], unsigned char> (&requirements)[5], const size_t offset, const __half *prob_to_cast) {
#else
__device__ __half get_casting_probability_2(const Lands& lands, const unsigned char cmc, const std::pair<unsigned char[8], unsigned char> (&requirements)[5], const size_t offset, const __half *prob_to_cast) {
#endif
    const auto counts = sum_with_mask_2(lands, requirements[0].first, requirements[1].first);
    return get_prob_to_cast(prob_to_cast, offset, std::get<0>(counts), std::get<1>(counts), std::get<2>(counts));
}

#ifdef CONSIDER_NON_BASICS
__device__ __half get_casting_probability_3(const Lands& lands, const unsigned char cmc, const std::pair<unsigned char[NUM_COMBINATIONS], unsigned char> (&requirements)[5], const size_t offset, const __half *prob_to_cast) {
#else
__device__ __half get_casting_probability_3(const Lands& lands, const unsigned char cmc, const std::pair<unsigned char[8], unsigned char> (&requirements)[5], const size_t offset, const __half *prob_to_cast) {
#endif
    unsigned char total_devotion = 0;
    __half probability = __float2half(1.f);
    for (size_t i=0; i < 3; i++) {
        const auto& color = requirements[i].first;
        const auto& required = requirements[i].second;
        total_devotion += required;
        probability *= get_prob_to_cast(prob_to_cast, cmc, required, sum_with_mask(lands, color));
    }
    unsigned char land_count = sum_with_mask_3(lands, requirements);
    return probability * get_prob_to_cast(prob_to_cast, cmc, total_devotion, land_count);
}

#ifdef CONSIDER_NON_BASICS
__device__ __half get_casting_probability_4(const Lands& lands, const unsigned char cmc, const std::pair<unsigned char[NUM_COMBINATIONS], unsigned char> (&requirements)[5], const size_t offset, const __half *prob_to_cast) {
#else
__device__ __half get_casting_probability_4(const Lands& lands, const unsigned char cmc, const std::pair<unsigned char[8], unsigned char> (&requirements)[5], const size_t offset, const __half *prob_to_cast) {
#endif
    unsigned char total_devotion = 0;
    __half probability = __float2half(1.f);
    for (size_t i=0; i < 4; i++) {
        const auto& color = requirements[i].first;
        const auto& required = requirements[i].second;
        total_devotion += required;
        probability *= get_prob_to_cast(prob_to_cast, cmc, required, sum_with_mask(lands, color));
    }
    unsigned char land_count = sum_with_mask_4(lands, requirements);
    return probability * get_prob_to_cast(prob_to_cast, cmc, total_devotion, land_count);
}

#ifdef CONSIDER_NON_BASICS
__device__ __half get_casting_probability_5(const Lands& lands, const unsigned char cmc, const std::pair<unsigned char[NUM_COMBINATIONS], unsigned char> (&requirements)[5], const size_t offset, const __half *prob_to_cast) {
#else
__device__ __half get_casting_probability_5(const Lands& lands, const unsigned char cmc, const std::pair<unsigned char[8], unsigned char> (&requirements)[5], const size_t offset, const __half *prob_to_cast) {
#endif
    unsigned char total_devotion = 0;
    __half probability = __float2half(1.f);
    for (size_t i=0; i < 5; i++) {
        const auto& color = requirements[i].first;
        const auto& required = requirements[i].second;
        total_devotion += required;
        probability *= get_prob_to_cast(prob_to_cast, cmc, required, sum_with_mask(lands, color));
    }
    unsigned char land_count = sum_with_mask_5(lands, requirements);
    return probability * get_prob_to_cast(prob_to_cast, cmc, total_devotion, land_count);
}

__device__ __half get_casting_probability(const Lands& lands, unsigned char cmc, const ColorRequirement& requirements, const __half* prob_to_cast, const unsigned char requirement_count) {
    switch(requirement_count) {
        case 0:
        return get_casting_probability_0(lands, cmc, requirements.requirements,  requirements.offset, prob_to_cast);
        case 1:
        return get_casting_probability_1(lands, cmc,  requirements.requirements,  requirements.offset, prob_to_cast);
        case 2:
        return get_casting_probability_2(lands, cmc,  requirements.requirements,  requirements.offset, prob_to_cast);
        case 3:
        return get_casting_probability_3(lands, cmc,  requirements.requirements,  requirements.offset, prob_to_cast);
        case 4:
        return get_casting_probability_4(lands, cmc,  requirements.requirements,  requirements.offset, prob_to_cast);
        case 5:
        return get_casting_probability_5(lands, cmc,  requirements.requirements,  requirements.offset, prob_to_cast);
    }
    return 0;
}

__device__ __half calculate_synergy(const __half similarity, const HalfVariables& variables) {
    if (similarity > __int2half_rd(1)) return variables.equal_cards_synergy;
    const __half scaled = variables.similarity_multiplier * __float2half(max(0.f, __half2float(similarity - variables.similarity_clip)));
    const __half transformed = __float2half(1.f) / (__float2half(1.f) - scaled) - __float2half(1.f);
    if (ISNAN(__half2float(transformed))) return 0;
    else return __float2half(min(__half2float(transformed), 10.f));
}

__device__ __half rating_oracle(const index_type card_index, const Lands&, const HalfVariables& variables, const HalfExpandedPick& pick,
                    const __half probability) {
    return probability * variables.ratings[pick.in_pack[card_index]];
}

__device__ __half pick_synergy_oracle(const index_type, const Lands&, const HalfVariables&, const HalfExpandedPick& pick,
                                                const __half (&probabilities)[MAX_PICKED], const __half probability, const __half (&synergies)[MAX_PICKED]) {
    if (pick.picked_count == 0) return 0;
    __half total_synergy = 0;
    for (size_t i=0; i < pick.picked_count; i++) total_synergy += probabilities[i] * synergies[i];
    return total_synergy * probability / __int2half_rd(pick.picked_count);
}

__device__ __half fixing_oracle(const index_type card_index, const Lands& lands, const HalfVariables& variables, const HalfExpandedPick& pick) {
    if (pick.in_pack_is_land[card_index]) {
        __half overlap = 0;
        for (size_t i=0; i < NUM_COLORS; i++){
            if (pick.in_pack_card_colors[card_index][i]) {
#ifdef CONSIDER_NON_BASICS
                unsigned char count = 0;
                for (size_t j=0; j < 16; j++) count += lands[INCLUSION_MAP[i][j]];
#else
                const unsigned char count = lands[i + 1];
#endif
                if (count >= LANDS_TO_INCLUDE_COLOR) overlap += MAX_SCORE / 5;
            }
        }
        if (pick.in_pack_is_fetch[card_index]) return variables.is_fetch_multiplier * overlap;
        else if (pick.in_pack_has_basic_land_types[card_index]) return variables.has_basic_types_multiplier * overlap;
        else return variables.is_regular_land_multiplier * overlap;
    } else return 0;
}

__device__ __half internal_synergy_oracle(const index_type, const Lands&, const HalfVariables&, const HalfExpandedPick& pick,
                              const __half (&probabilities)[MAX_PICKED], const __half (&synergies)[MAX_PICKED][MAX_PICKED]) {
    if (pick.picked_count < 2) return 0;
    __half internal_synergy = 0;
    for(index_type i=0; i < pick.picked_count; i++) {
        if (probabilities[i] > __float2half(0.f)) {
            __half card_synergy = 0;
            for (index_type j = 0; j < i; j++) card_synergy += probabilities[j] * synergies[i][j];
            internal_synergy += probabilities[i] * card_synergy;
        }
    }
    return __float2half(2.f) * internal_synergy / __int2half_rd(pick.picked_count * (pick.picked_count - 1));
}

template<size_t Size>
__device__ __half sum_gated_rating(const HalfVariables& variables, const index_type (&indices)[Size],
                       const __half (&probabilities)[Size],
                       const index_type num_valid_indices) {
    __half result = 0;
    if (num_valid_indices == 0) return 0;
    for (index_type i=0; i < num_valid_indices; i++) result += variables.ratings[indices[i]] * probabilities[i];
    return result / (__half) num_valid_indices;
}

__device__ __half openness_oracle(const index_type, const Lands&, const HalfVariables& variables, const HalfExpandedPick& pick,
                                            const __half (&probabilities)[MAX_SEEN]) {
    return sum_gated_rating(variables, pick.seen, probabilities, pick.seen_count);
}

__device__ __half colors_oracle(const index_type, const Lands&, const HalfVariables& variables, const HalfExpandedPick& pick,
                                          const __half (&probabilities)[MAX_PICKED]) {
    return sum_gated_rating(variables, pick.picked, probabilities, pick.picked_count);
}

__device__ __half get_score(const index_type card_index, const Lands& lands, const HalfVariables& variables, const HalfExpandedPick& pick,
                const __half rating_weight, const __half pick_synergy_weight, const __half fixing_weight,
                const __half internal_synergy_weight, const __half openness_weight, const __half colors_weight,
                const __half* prob_to_cast,
                const __half (&internal_synergies)[MAX_PICKED][MAX_PICKED],
                const __half (&pick_synergies)[MAX_PICKED],
                const unsigned char card_requirements, const unsigned char (&seen_requirements)[MAX_SEEN], const unsigned char (&picked_requirements)[MAX_PICKED]) { //, const ColorRequirement& card_color_requirement,
                /* const ColorRequirement (&seen_color_requirements)[MAX_SEEN], const ColorRequirement (&picked_color_requirements)[MAX_PICKED]) { */
    __half seen_probabilities[MAX_SEEN];
    __half picked_probabilities[MAX_PICKED];
    for (index_type i=0; i < pick.seen_count; i++) {
        seen_probabilities[i] =
                __float2half(max(__half2float(get_casting_probability(lands, pick.seen_cmcs[i], pick.seen_color_requirements[i], prob_to_cast, seen_requirements[i]) - variables.prob_to_include), 0.f))
                * variables.prob_multiplier;
    }
    for (index_type i=0; i < pick.picked_count; i++) {
        picked_probabilities[i] =
                __float2half(max(__half2float(get_casting_probability(lands, pick.picked_cmcs[i], pick.picked_color_requirements[i], prob_to_cast, picked_requirements[i]) - variables.prob_to_include), 0.f))
                * variables.prob_multiplier;
    }
    const __half card_casting_probability = get_casting_probability(lands, pick.in_pack_cmcs[card_index], pick.in_pack_color_requirements[card_index], prob_to_cast, card_requirements);
    const __half rating_score = rating_oracle(card_index, lands, variables, pick, card_casting_probability);
    const __half pick_synergy_score = pick_synergy_oracle(card_index, lands, variables, pick, picked_probabilities,
                                                          card_casting_probability, pick_synergies);
    const __half fixing_score = fixing_oracle(card_index, lands, variables, pick);
    const __half internal_synergy_score = internal_synergy_oracle(card_index, lands, variables, pick,
                                                                  picked_probabilities, internal_synergies);
    const __half openness_score = openness_oracle(card_index, lands, variables, pick, seen_probabilities);
    const __half colors_score = colors_oracle(card_index, lands, variables, pick, picked_probabilities);
    return rating_score*rating_weight + pick_synergy_score*pick_synergy_weight + fixing_score*fixing_weight
           + internal_synergy_score*internal_synergy_weight + openness_score*openness_weight + colors_score*colors_weight;
}

__device__ __half do_climb(const index_type card_index, const HalfVariables& variables, const HalfExpandedPick& pick, const __half* prob_to_cast,
               const __half (&internal_synergies)[MAX_PICKED][MAX_PICKED]) {
    __half previous_score = -1;
    __half current_score = 0;
    const __half rating_weight = interpolate_weights(variables.rating_weights, pick);
    const __half pick_synergy_weight = interpolate_weights(variables.pick_synergy_weights, pick);
    const __half fixing_weight = interpolate_weights(variables.fixing_weights, pick);
    const __half internal_synergy_weight = interpolate_weights(variables.internal_synergy_weights, pick);
    const __half openness_weight = interpolate_weights(variables.openness_weights, pick);
    const __half colors_weight = interpolate_weights(variables.colors_weights, pick);
    __half pick_synergies[MAX_PICKED];
    const unsigned char card_requirements = pick.in_pack_color_requirements[card_index].requirements_count;
    unsigned char seen_requirements[MAX_SEEN];
    for (index_type i=0; i < pick.seen_count; i++) seen_requirements[i] = pick.seen_color_requirements[i].requirements_count;
    unsigned char picked_requirements[MAX_PICKED];
    for (index_type i=0; i < pick.picked_count; i++) picked_requirements[i] = pick.picked_color_requirements[i].requirements_count;
    /* ColorRequirement card_color_requirement = pick.in_pack_color_requirements[card_index]; */
    /* ColorRequirement seen_color_requirements[MAX_SEEN]; */
    /* for (size_t i=0; i < pick.seen_count; i++) seen_color_requirements[i] = pick.seen_color_requirements[i]; */
    /* ColorRequirement picked_color_requirements[MAX_PICKED]; */
    /* for (size_t i=0; i < pick.picked_count; i++) picked_color_requirements[i] = pick.picked_color_requirements[i]; */
    for (index_type i=0; i < pick.picked_count; i++) pick_synergies[i] = calculate_synergy(pick.picked_similarities[card_index][i], variables);
    Lands lands{0, 4, 4, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    while (previous_score < current_score) {
        previous_score = current_score;
        for(size_t remove_index=1; remove_index < COLORS.size() + 1; remove_index++) {
            while (remove_index < COLORS.size() + 1 && lands[remove_index] == 0) remove_index++;
            if (remove_index < COLORS.size() + 1) {
                bool breakout = false;
                for (size_t add_index=1; add_index < COLORS.size() + 1; add_index++) {
                    if (add_index == remove_index) add_index++;
                    if (add_index < COLORS.size() + 1) {
                        Lands new_lands{0};
                        for (size_t i=0; i < NUM_COMBINATIONS; i++) new_lands[i] = lands[i];
                        new_lands[remove_index] -= 1;
                        new_lands[add_index] += 1;
                        __half score = get_score(card_index, new_lands, variables, pick, rating_weight, pick_synergy_weight,
                                                fixing_weight, internal_synergy_weight, openness_weight, colors_weight,
                                                prob_to_cast, internal_synergies, pick_synergies,
                                                card_requirements, seen_requirements, picked_requirements); //, card_color_requirement,
                                                /* seen_color_requirements, picked_color_requirements); */
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

__device__ std::pair<double, bool> calculate_loss(const HalfExpandedPick& pick, const HalfVariables& variables, const double temperature, const __half* prob_to_cast) {
    double scores[MAX_PACK_SIZE]{0};
    __half internal_synergies[MAX_PICKED][MAX_PICKED]{0};
    for (index_type i=0; i < pick.picked_count; i++) {
        for (index_type j=0; j < i; j++) {
            internal_synergies[i][j] = calculate_synergy(pick.picked_similarities[i][j], variables);
        }
    }
    double max_score = 0;
    double denominator = 0;
    for (index_type i=0; i < pick.in_pack_count; i++) {
        const double score = do_climb(i, variables, pick, prob_to_cast, internal_synergies);
        scores[i] = EXP(__half2float(score) / temperature);
        max_score = max(scores[i], max_score);
        denominator += scores[i];
    }
    bool best = true;
    for (index_type i=0; i < pick.in_pack_count; i++) best &= (pick.in_pack[i] == pick.chosen_card) == (scores[i] == max_score);
    for(index_type i=0; i < pick.in_pack_count; i++) {
        if (pick.in_pack[i] == pick.chosen_card) {
            if (scores[i] >= 0) {
                return {-LOG(scores[i] / denominator), best};
            } else {
                return {-1, false};
            }
        }
    }
    return {-2, false};
}

__global__ void run_thread(const __half* prob_to_cast_ptr, const HalfVariables* variables_ptr, const HalfExpandedPick* picks_ptr, const double temperature,
                           double* loss_results_ptr, bool* accuracy_results_ptr, const size_t variables_offset) {
    size_t variable_id = variables_offset + blockIdx.x;
    size_t initial_pick_id = blockDim.x * blockIdx.y + threadIdx.x;
    size_t pick_stride = blockDim.x * gridDim.y;
    __shared__ HalfVariables variables;
    if (threadIdx.x == 0) variables = variables_ptr[variable_id];
    __syncthreads();
    for (size_t i=initial_pick_id; i < PICKS_PER_GENERATION; i += pick_stride) {
        std::pair<double, bool> result = calculate_loss(picks_ptr[i], variables, temperature, prob_to_cast_ptr);
        loss_results_ptr[variable_id * PICKS_PER_GENERATION + i] = result.first;
        accuracy_results_ptr[variable_id * PICKS_PER_GENERATION + i] = result.second;
    }
}

std::array<std::array<double, 4>, POPULATION_SIZE> run_simulations_cuda(const std::vector<Variables>& variables,
                                                                        const std::vector<ExpandedPick>& picks, const float temperature,
                                                                        const float (&prob_to_cast)[PROB_DIM]) {
    std::vector<HalfVariables> half_variables;
    half_variables.reserve(variables.size());
    for (const Variables& variable : variables) half_variables.emplace_back(variable);
    std::vector<HalfExpandedPick> half_picks;
    half_picks.reserve(picks.size());
    for (const ExpandedPick& pick : picks) half_picks.emplace_back(pick);
    __half* prob_to_cast_ptr;
    HalfVariables* variables_ptr;
    HalfExpandedPick* picks_ptr;
    double *loss_results_ptr;
    bool* accuracy_results_ptr;
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);
    std::cout << "Sizeof HalfVariables: " << sizeof(HalfVariables) << ", HalfExpandedPick: " << sizeof(HalfExpandedPick) << ", ProbToCast: " << sizeof(__half[PROB_DIM]) << std::endl;
    std::cout << "cooperativeMultiDeviceLaunch: " << props.cooperativeMultiDeviceLaunch << " maxBlocksPerMultiProcessor: "
              << props.maxBlocksPerMultiProcessor << " maxThreadsPerMultiProcessor: " << props.maxThreadsPerMultiProcessor
              << " maxThreadsDim[0]: " << props.maxThreadsDim[0] << " maxThreadsDim[1]: " << props.maxThreadsDim[1]
              << " maxThreadsDim[2]: " << props.maxThreadsDim[2] << " maxThreadsPerBlock: " << props.maxThreadsPerBlock
              << " multiProcessorCount: " << props.multiProcessorCount << " warpSize: " << props.warpSize << std::endl;
    cudaMallocManaged(&prob_to_cast_ptr, sizeof(__half[PROB_DIM]));
    for (size_t i=0; i < PROB_DIM; i++) prob_to_cast_ptr[i] = __float2half(prob_to_cast[i]);
    cudaMallocManaged(&variables_ptr, sizeof(HalfVariables) * half_variables.size());
    cudaMemcpy(variables_ptr, half_variables.data(), sizeof(HalfVariables) * half_variables.size(), cudaMemcpyHostToHost);
    cudaMallocManaged(&picks_ptr, sizeof(HalfExpandedPick) * half_picks.size());
    cudaMemcpy(picks_ptr, half_picks.data(), sizeof(HalfExpandedPick) * half_picks.size(), cudaMemcpyHostToHost);
    cudaMallocManaged(&loss_results_ptr, sizeof(double) * variables.size() * picks.size());
    cudaMallocManaged(&accuracy_results_ptr, sizeof(bool) * variables.size() * picks.size());
    int min_grid_size;
    int block_size;
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, run_thread, 0, PICKS_PER_GENERATION);
    std::cout << "Min Grid Size: " << min_grid_size << ", Block Size: " << block_size << std::endl;
    const dim3 gridDim(variables.size(), (props.maxThreadsPerMultiProcessor / block_size) * min_grid_size / variables.size());
    const dim3 blockDim(block_size);
    run_thread<<<gridDim, blockDim>>>(prob_to_cast_ptr, variables_ptr, picks_ptr, temperature,
                                      loss_results_ptr, accuracy_results_ptr, 0);
    cudaDeviceSynchronize();
    std::array<std::array<double, 4>, POPULATION_SIZE> results;
    for (size_t i = 0; i < POPULATION_SIZE; i++) {
        size_t count_correct = 0;
        for (size_t j = 0; j < PICKS_PER_GENERATION; j++) {
            results[i][1] += loss_results_ptr[i * picks.size() + j];
            if (accuracy_results_ptr[i * picks.size() + j]) count_correct++;
        }
        results[i][3] = count_correct / (double) PICKS_PER_GENERATION;
        results[i][2] = -LOG(results[i][3]);
        results[i][1] /= PICKS_PER_GENERATION;
        results[i][0] = CATEGORICAL_CROSSENTROPY_LOSS_WEIGHT * results[i][1]
                        + NEGATIVE_LOG_ACCURACY_LOSS_WEIGHT * results[i][2];
    }
    cudaFree(prob_to_cast_ptr);
    cudaFree(variables_ptr);
    cudaFree(picks_ptr);
    cudaFree(loss_results_ptr);
    cudaFree(accuracy_results_ptr);
    return results;
}
