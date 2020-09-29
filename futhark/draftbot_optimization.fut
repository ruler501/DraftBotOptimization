let PACKS: i8 = 3
let PACK_SIZE: i8 = 15
let MAX_SCORE: f32 = 10
let NUM_COLORS: i8 = 5
let NUM_COMBINATIONS: i8 = 5
let CATEGORICAL_CROSSENTROPY_LOSS_WEIGHT: f64 = 0
let NEGATIVE_LOG_ACCURACY_LOSS_WEIGHT: f64 = 1

type Colors = [5]bool
type LandCounts = [5]i8
type ColorRequirement = {masks: LandCounts, count: i8}
type ProbTable = [16][9][4][18][18]f32
type ColorRequirements = #no_requirements i8 | #one_requirement i8 ColorRequirement | #two_requirements i8 ColorRequirement ColorRequirement
                       | #three_requirements i8 ColorRequirement ColorRequirement ColorRequirement
                       | #four_requirements i8 ColorRequirement ColorRequirement ColorRequirement ColorRequirement
                       | #five_requirements i8 ColorRequirement ColorRequirement ColorRequirement ColorRequirement ColorRequirement
type Weights = [3][15]f32
type Ratings = [21467]f32
type Variables = {
    rating_weights: Weights,
    pick_synergy_weights: Weights,
    fixing_weights: Weights,
    internal_synergy_weights: Weights,
    openness_weights: Weights,
    colors_weights: Weights,
    ratings: Ratings,
    prob_to_include: f32,
    prob_multiplier: f32,
    similarity_clip: f32,
    similarity_multiplier: f32,
    is_fetch_multiplier: f32,
    has_basic_types_multiplier: f32,
    is_regular_land_multiplier: f32, 
    equal_cards_synergy: f32
}
type Pick = {
    in_pack_count: i8,
    in_pack: [20]i16,
    in_pack_color_requirements: [20]ColorRequirements,
    in_pack_is_land: [20]bool,
    in_pack_is_fetch: [20]bool,
    in_pack_has_basic_land_types: [20]bool,
    in_pack_colors: [20]Colors,
    in_pack_picked_similarities: [20][64]f32,
    picked_count: i8,
    picked: [64]i16,
    picked_color_requirements: [64]ColorRequirements,
    internal_similarities: [64][64]f32,
    seen_count: i16,
    seen: [408]i16,
    seen_color_requirements: [408]ColorRequirements,
    pack_num: i8,
    packs_count: i8,
    pick_num: i8,
    pack_size: i8,
    chosen_index: i8
}

let DEFAULT_LANDS: LandCounts = [4, 4, 3, 3, 3]

let sum_with_mask (counts: LandCounts) (masks: LandCounts) = reduce (+) 0 (map2 (&) counts masks)

let sum_with_mask_n (counts: LandCounts) (masks: []LandCounts) = sum_with_mask counts (reduce (\a m -> map2 (|) a m) (replicate 5 0i8) masks)

let get_casting_probability_0 : f32 = 0

let get_casting_probability_1 (lands: LandCounts) (cmc: i8) (requirement: ColorRequirement) (prob_to_cast: ProbTable) =
    prob_to_cast[cmc, requirement.count, 0, sum_with_mask lands requirement.masks, 0]

let get_casting_probability_2 (lands: LandCounts) (cmc: i8) (req1: ColorRequirement) (req2: ColorRequirement)
                              (prob_to_cast: ProbTable) =
    prob_to_cast[cmc, req1.count, req2.count, sum_with_mask lands req1.masks, sum_with_mask lands req2.masks]

let get_casting_probability_n (lands: LandCounts) (cmc: i8) (reqs: []ColorRequirement) (prob_to_cast: ProbTable) =
    reduce (*) prob_to_cast[cmc, reduce (+) 0 (map (.count) reqs), 0, sum_with_mask_n lands (map (.masks) reqs), 0]
        (map (\req -> get_casting_probability_1 lands cmc req prob_to_cast) reqs)

let get_casting_probability (lands: LandCounts) (requirements: ColorRequirements) (prob_to_cast: ProbTable) =
    match requirements
    case #no_requirements -> get_casting_probability_0
    case #one_requirement cmc req1 -> get_casting_probability_1 lands cmc req1 prob_to_cast
    case #two_requirements cmc req1 req2 -> get_casting_probability_2 lands cmc req1 req2 prob_to_cast
    case #three_requirements cmc req1 req2 req3 -> get_casting_probability_n lands cmc [req1, req2, req3] prob_to_cast
    case #four_requirements cmc req1 req2 req3 req4 -> get_casting_probability_n lands cmc [req1, req2, req3, req4] prob_to_cast
    case #five_requirements cmc req1 req2 req3 req4 req5 -> get_casting_probability_n lands cmc [req1, req2, req3, req4, req5] prob_to_cast

let interpolate_weights (weights: Weights) (pack_num: i8) (packs_count: i8) (pick_num: i8) (pack_size: i8) =
    let x_index = f32.i8(PACKS) * f32.i8(pack_num) / f32.i8(packs_count)
    let y_index = f32.i8(PACK_SIZE) * f32.i8(pick_num) / f32.i8(pack_size)
    let floor_x_index = i8.f32(x_index)
    let floor_y_index = i8.f32(y_index)
    let ceil_x_index = i8.min (floor_x_index + 1) (PACKS - 1)
    let ceil_y_index = i8.min (floor_y_index + 1) (PACK_SIZE - 1)
    let x_index_mod_one = x_index - f32.i8(floor_x_index)
    let y_index_mod_one = y_index - f32.i8(floor_y_index)
    let inv_x_index_mod_one = 1 - x_index_mod_one
    let inv_y_index_mod_one = 1 - y_index_mod_one
    let XY = x_index_mod_one * y_index_mod_one
    let Xy = x_index_mod_one * inv_y_index_mod_one
    let xY = inv_x_index_mod_one * y_index_mod_one
    let xy = inv_x_index_mod_one * inv_y_index_mod_one
    let XY_weight = weights[ceil_x_index, ceil_y_index]
    let Xy_weight = weights[ceil_x_index, floor_y_index]
    let xY_weight = weights[floor_x_index, ceil_y_index]
    let xy_weight = weights[floor_x_index, floor_y_index]
    in XY * XY_weight + Xy * Xy_weight + xY * xY_weight + xy * xy_weight

let calculate_probability (lands: LandCounts) (requirements: ColorRequirements) (prob_to_cast: ProbTable) (prob_multiplier: f32) (prob_to_include: f32) =
    prob_multiplier * (f32.max ((get_casting_probability lands requirements prob_to_cast) - prob_to_include) 0)

let calculate_synergy (similarity: f32) (similarity_clip: f32) (similarity_multiplier: f32) =
    if similarity > 1 then 0
    else
        let scaled = similarity_multiplier * f32.max 0 (similarity - similarity_clip)
        let transformed = (1 / (1 - scaled)) - 1
        in f32.min transformed MAX_SCORE

let rating_oracle (card_index: i16) (probability: f32) (ratings: Ratings) = 
    probability * ratings[card_index]

let pick_synergy_oracle [picked_count] (probability: f32) (probabilities: [picked_count]f32) (synergies: [picked_count]f32) =
    if picked_count == 0 then 0
    else
        probability * (reduce (+) 0 (map2 (*) probabilities synergies)) / f32.i32(picked_count)

let fixing_oracle (lands: LandCounts) (is_land: bool) (is_fetch: bool) (has_basic_land_types: bool)
                  (colors: Colors) (is_fetch_multiplier: f32) (has_basic_types_multiplier: f32) (is_regular_land_multiplier: f32) =
    if is_land then
        let overlap = (MAX_SCORE / 5) * (reduce (+) 0 (map2 (\l c -> if (l >= 3) && c then 1 else 0) lands colors))
        let multiplier = if is_fetch then is_fetch_multiplier
            else if has_basic_land_types then has_basic_types_multiplier
                else is_regular_land_multiplier
        in overlap * multiplier
    else 0

let internal_synergy_oracle [picked_count] (probabilities: [picked_count]f32) (synergies: [picked_count][picked_count]f32) =
    if picked_count < 2 then 0
    else
        reduce (+) 0 (map (\i -> probabilities[i] * (reduce (+) 0 (map2 (*) probabilities[:i] synergies[i, :i]))) (indices probabilities))

let sum_gated_rating [count] (ratings: Ratings) (indices: [count]i16) (probabilities: [count]f32) =
    if count == 0 then 0
    else
        (reduce (+) 0 (map2 (\p i -> p * ratings[i]) probabilities indices)) / f32.i32(count)

let openness_oracle = sum_gated_rating

let colors_oracle = sum_gated_rating

let get_score [picked_count] [seen_count] (lands: LandCounts) (variables: Variables) (rating_weight: f32)
              (pick_synergy_weight: f32) (fixing_weight: f32) (internal_synergy_weight: f32) (openness_weight: f32)
              (colors_weight: f32) (prob_to_cast: ProbTable) (card_index: i16) (card_color_requirements: ColorRequirements)
              (card_is_land: bool) (card_is_fetch: bool) (card_has_basic_land_types: bool) (card_colors: Colors)
              (card_synergies: [picked_count]f32) (picked: [picked_count]i16) (picked_color_requirements: [picked_count]ColorRequirements)
              (internal_synergies: [picked_count][picked_count]f32) (seen: [seen_count]i16) (seen_color_requirements: [seen_count]ColorRequirements) =
    let card_probability = calculate_probability lands card_color_requirements prob_to_cast variables.prob_multiplier variables.prob_to_include
    let picked_probabilities = map (\req -> calculate_probability lands req prob_to_cast variables.prob_multiplier variables.prob_to_include) picked_color_requirements
    let seen_probabilities = map (\req -> calculate_probability lands req prob_to_cast variables.prob_multiplier variables.prob_to_include) seen_color_requirements
    let rating_score = rating_oracle card_index card_probability variables.ratings
    let pick_synergy_score = pick_synergy_oracle card_probability picked_probabilities card_synergies
    let fixing_score = fixing_oracle lands card_is_land card_is_fetch card_has_basic_land_types card_colors variables.is_fetch_multiplier
                                     variables.has_basic_types_multiplier variables.is_regular_land_multiplier
    let internal_synergy_score = internal_synergy_oracle picked_probabilities internal_synergies
    let openness_score = openness_oracle variables.ratings seen seen_probabilities
    let colors_score = colors_oracle variables.ratings picked picked_probabilities
    in rating_score * rating_weight + pick_synergy_score * pick_synergy_weight + fixing_score * fixing_weight
        + internal_synergy_score * internal_synergy_weight + openness_score * openness_weight
        + colors_score * colors_weight

let do_climb [picked_count] [seen_count] (variables: Variables) (card_index: i16) (card_color_requirements: ColorRequirements)
             (card_is_land: bool) (card_is_fetch: bool) (card_has_basic_land_types: bool) (card_colors: Colors)
             (card_similarities: [picked_count]f32) (picked: [picked_count]i16) (picked_color_requirements: [picked_count]ColorRequirements)
             (internal_synergies: [picked_count][picked_count]f32) (seen: [seen_count]i16) (seen_color_requirements: [seen_count]ColorRequirements)
             (prob_to_cast: ProbTable) (rating_weight: f32) (pick_synergy_weight: f32) (fixing_weight: f32)
             (internal_synergy_weight: f32) (openness_weight: f32) (colors_weight: f32) =
    let card_synergies = map (\sim -> calculate_synergy sim variables.similarity_clip variables.similarity_multiplier) card_similarities
    let (_, _, final_score) = loop (lands, previous_score, current_score) = (DEFAULT_LANDS, -1, 0) while previous_score < current_score do
        let (next_lands, next_score, _) = loop (new_lands, new_score, remove) = (lands, current_score, 0) while (new_score <= current_score) && (remove < NUM_COLORS) do
            if lands[remove] <= 0 then (new_lands, new_score, remove + 1)
            else
                let (n_lands, n_score, _) = loop (n2_lands, n2_score, add) = ((map (\x -> x) lands) with [remove] = lands[remove] - 1, current_score, 0) while (n2_score <= current_score) && (add < NUM_COLORS) do
                    if remove == add then (n2_lands, n2_score, add + 1)
                    else 
                        let modified_lands = (map (\x -> x) n2_lands) with [add] = lands[add] + 1
                        let score = get_score modified_lands variables rating_weight pick_synergy_weight
                                              fixing_weight internal_synergy_weight openness_weight colors_weight
                                              prob_to_cast card_index card_color_requirements card_is_land
                                              card_is_fetch card_has_basic_land_types card_colors card_synergies
                                              picked picked_color_requirements internal_synergies seen seen_color_requirements
                        in if score > current_score then (modified_lands, score, add + 1)
                        else (n2_lands, n2_score, add + 1)
                in if n_score > current_score then (n_lands, n_score, remove + 1)
                else (new_lands, new_score, remove + 1)
        in (next_lands, current_score, next_score)
    in final_score

let calculate_loss (pick: Pick) (variables: Variables) (prob_to_cast: ProbTable) (temperature: f64) =
    let rating_weight = interpolate_weights variables.rating_weights pick.pack_num pick.packs_count pick.pick_num pick.pack_size
    let pick_synergy_weight = interpolate_weights variables.pick_synergy_weights pick.pack_num pick.packs_count pick.pick_num pick.pack_size
    let fixing_weight = interpolate_weights variables.fixing_weights pick.pack_num pick.packs_count pick.pick_num pick.pack_size
    let internal_synergy_weight = interpolate_weights variables.internal_synergy_weights pick.pack_num pick.packs_count pick.pick_num pick.pack_size
    let openness_weight = interpolate_weights variables.openness_weights pick.pack_num pick.packs_count pick.pick_num pick.pack_size
    let colors_weight = interpolate_weights variables.colors_weights pick.pack_num pick.packs_count pick.pick_num pick.pack_size
    let picked_count = i32.i8(pick.picked_count)
    let internal_synergies = map (\row: [picked_count]f32 -> map (\x -> calculate_synergy x variables.similarity_clip variables.similarity_multiplier) row[:picked_count]) pick.internal_similarities[:picked_count]
    let pick_synergies = map (\row: [picked_count]f32 -> map (\x -> calculate_synergy x variables.similarity_clip variables.similarity_multiplier) row[:picked_count]) pick.in_pack_picked_similarities[:i32.i8(pick.in_pack_count)]
    let scores = map (\i -> do_climb variables pick.in_pack[i] pick.in_pack_color_requirements[i] pick.in_pack_is_land[i]
                                     pick.in_pack_is_fetch[i] pick.in_pack_has_basic_land_types[i] pick.in_pack_colors[i]
                                     pick_synergies[i] pick.picked[:picked_count] pick.picked_color_requirements[:picked_count]
                                     internal_synergies pick.seen[:i32.i16(pick.seen_count)] pick.seen_color_requirements[:i32.i16(pick.seen_count)]
                                     prob_to_cast rating_weight pick_synergy_weight fixing_weight internal_synergy_weight
                                     openness_weight colors_weight) (iota (i32.i8(pick.in_pack_count)))
    let exp_scores = map f64.exp (map (/temperature) (map f64.f32 scores))
    let sum_exp_scores = reduce (+) 0 exp_scores
    let chosen_score = scores[pick.chosen_index]
    in (-f64.log (exp_scores[pick.chosen_index] / sum_exp_scores),
        if reduce (&&) true (map (\i -> i == i32.i8(pick.chosen_index) || scores[i] < chosen_score) (indices scores)) then 1f64 else 0f64)

let calculate_batch_loss [n] (picks: [n]Pick) (variables: Variables) (prob_to_cast: ProbTable) (temperature: f64) =
    let (cross_entropies, accuracies) = unzip (map (\p -> calculate_loss p variables prob_to_cast temperature) picks)
    let accuracy = (reduce (+) 0 accuracies) / f64.i32(n)
    let neg_log_accuracy = -f64.log accuracy
    let cross_entropy = (reduce (+) 0 cross_entropies) / f64.i32(n)
    let loss = CATEGORICAL_CROSSENTROPY_LOSS_WEIGHT * cross_entropy + NEGATIVE_LOG_ACCURACY_LOSS_WEIGHT * neg_log_accuracy
    in [loss, cross_entropy, neg_log_accuracy, accuracy]

entry run_simulations [n] [m] (picks: [n]Pick) (variables: [m]Variables) (prob_to_cast: ProbTable) (temperature: f64) =
    map (\v -> calculate_batch_loss picks v prob_to_cast temperature) variables
