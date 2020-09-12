//
// Created by Devon Richards on 9/6/2020.
//
#include <array>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <vector>

#include <cuda_runtime_api.h>

// #include <Eigen/Dense>
#include <magma_v2.h>

#include "../draftbot_optimization.h"
#include "shared/parameters.h"

Variables optimize_variables(const float temperature, const std::vector<Pick>& picks, const size_t num_generations,
                             const std::shared_ptr<const Constants>& constants,
                             const std::shared_ptr<const Variables>& initial_variables, const size_t seed) {
    std::cout << "Beginning optimize_variables with population size of " << POPULATION_SIZE << " and " << picks.size()
              << " picks, sampling " << PICKS_PER_GENERATION << " per generation" << std::endl << std::endl;
    std::mt19937_64 gen{seed};
    std::normal_distribution<float> unit_normal(0, 1);
    std::uniform_int_distribution<size_t> pick_selector(0, picks.size() - 1);
    std::cout << __LINE__ << std::endl;
    cudaSetDeviceFlags(cudaDeviceBlockingSync);
    magma_init();
    magma_queue_t queue=nullptr;
    magma_device_t devices[2];
    magma_int_t device_count;
    magma_getdevices(devices, 2, &device_count);
    std::cout << device_count << " devices first of which has id " << devices[0] << std::endl;
    magma_queue_create(0, &queue);
    const magma_int_t rounded_dimension = magma_roundup(Variables::num_parameters, 32);
    const magma_int_t dimension = Variables::num_parameters;
    magmaDouble_ptr covariance = nullptr;
    magmaDouble_ptr mean = nullptr;
    magmaDouble_ptr iso_evolution_path = nullptr;
    magmaDouble_ptr aniso_evolution_path = nullptr;
    magma_dmalloc(&covariance, rounded_dimension * dimension);
    magma_dmalloc(&mean, dimension);
    magma_dmalloc(&iso_evolution_path, dimension);
    magma_dmalloc(&aniso_evolution_path, dimension);
    magmablas_dlaset(MagmaFull, dimension, dimension, 0, 1, covariance, rounded_dimension, queue);
    magmablas_dlaset(MagmaFull, dimension, 1, 0, 0, iso_evolution_path, dimension, queue);
    magmablas_dlaset(MagmaFull, dimension, 1, 0, 0, aniso_evolution_path, dimension, queue);
    std::cout << __LINE__ << std::endl;
    float step_size = INITIAL_STD_DEV;
    const std::array<float, POPULATION_SIZE> PRELIMINARY_WEIGHTS = [] {
        std::array<float, POPULATION_SIZE> result{};
        for (size_t i = 0; i < POPULATION_SIZE; i++) result[i] = std::log(KEEP_BEST + 0.5f) - std::log((float)i + 1);
        return result;
    }();
    const float sum_preliminary_weights_positive = std::accumulate(PRELIMINARY_WEIGHTS.begin(),
                                                                   PRELIMINARY_WEIGHTS.begin() + KEEP_BEST, 0.f);
    const float variance_selection_mass = std::pow(sum_preliminary_weights_positive, 2)
                                           / std::accumulate(PRELIMINARY_WEIGHTS.begin(),
                                                             PRELIMINARY_WEIGHTS.begin() + KEEP_BEST,
                                                            0.f, [](float w1, float w2){ return w1 + w2 * w2; });
    const float sum_preliminary_weights_negative = std::accumulate(PRELIMINARY_WEIGHTS.begin() + KEEP_BEST,
                                                                   PRELIMINARY_WEIGHTS.end(), 0.f);
    const float negative_selection_mass = std::pow(sum_preliminary_weights_negative, 2)
                                            / std::accumulate(PRELIMINARY_WEIGHTS.begin() + KEEP_BEST,
                                                              PRELIMINARY_WEIGHTS.end(), 0.f,
                                                              [](float w1, float w2){ return w1 + w2 * w2; });
    const float rank_one_learning_rate = ALPHA_COVARIANCE / (std::pow(Variables::num_parameters + 1.3f, 2.f) + variance_selection_mass);
    const float rank_u_learning_rate = std::min(1.f - rank_one_learning_rate,
                                                  ALPHA_COVARIANCE * (variance_selection_mass - 2
                                                                             + 1 / variance_selection_mass)
                                                  / (std::pow(Variables::num_parameters + 2.f, 2.f)
                                                        + ALPHA_COVARIANCE * variance_selection_mass / 2));
    const float iso_discount_factor = (variance_selection_mass + 2) / (Variables::num_parameters + variance_selection_mass + 5);
    const float iso_damping_factor = 1 + 2 * std::max(0.f, std::sqrt((variance_selection_mass - 1) / (Variables::num_parameters + 1)) - 1)
                                        + iso_discount_factor;
    const float aniso_discount_factor = (4 + variance_selection_mass / Variables::num_parameters)
                                            / (Variables::num_parameters + 4 + 2 * variance_selection_mass / Variables::num_parameters);
    const float alpha_mu = 1 + rank_one_learning_rate / rank_u_learning_rate;
    const float alpha_variance_mass = 1 + 2 * negative_selection_mass / (variance_selection_mass + 2);
    const float alpha_pos_def = (1 - rank_one_learning_rate - rank_u_learning_rate)
                                    / Variables::num_parameters / rank_u_learning_rate;
    std::cout << "Negative selection mass: " << negative_selection_mass << std::endl;
    const float negative_scaling = std::min(std::min(alpha_mu, alpha_variance_mass), alpha_pos_def) / -sum_preliminary_weights_negative;
    const std::array<float, POPULATION_SIZE> KEEP_WEIGHTS = [&] {
        std::array<float, POPULATION_SIZE> result{};
        for (size_t i = 0; i < KEEP_BEST; i++) result[i] = PRELIMINARY_WEIGHTS[i] / sum_preliminary_weights_positive;
        for (size_t i = KEEP_BEST; i < POPULATION_SIZE; i++) result[i] =  PRELIMINARY_WEIGHTS[i] * negative_scaling;
        return result;
    }();
    const float sum_keep_weights = std::accumulate(KEEP_WEIGHTS.begin(), KEEP_WEIGHTS.end(), 0.f);
    std::cout << "Sum of KEEP_WEIGHTS: " << sum_keep_weights << std::endl;
    std::cout << "variance_selection_mass: " << variance_selection_mass << std::endl;
    std::cout << "KEEP_WEIGHTS: ";
    for (size_t i=0; i < POPULATION_SIZE; i++) std::cout << KEEP_WEIGHTS[i] << ", ";
    std::cout << std::endl;
    const float expected_unit_norm = (float)std::sqrt(Variables::num_parameters)
                                     * (1 - 1 / 4.f / Variables::num_parameters
                                        + 1 / 21.f / Variables::num_parameters / Variables::num_parameters);

    std::vector<Variables> population(POPULATION_SIZE);
    std::vector<Pick> chosen_picks(PICKS_PER_GENERATION);
    std::array<std::array<double, 4>, POPULATION_SIZE> losses{};
    std::array<std::pair<size_t, double>, POPULATION_SIZE> indexed_losses{};
    std::array<std::pair<size_t, double>, POPULATION_SIZE> indexed_accuracies{};
    std::array<float, Variables::num_parameters> params{};
    std::array<float, POPULATION_SIZE> normalized_weights{};
    std::array<float, Variables::num_parameters> inverse_eigenvalues{};
    std::array<magmaDouble_ptr, POPULATION_SIZE> changes{};
    magmaDouble_ptr eigenvalues = nullptr;
    magmaDouble_ptr zs = nullptr;
    magmaDouble_ptr temporary_x = nullptr;
    magmaDouble_ptr change_in_mean = nullptr;
    magmaDouble_ptr inverse_root_times_change_in_mean = nullptr;
    magmaDouble_ptr max_coordinates = nullptr;
    magmaDouble_ptr eigenvalues_matrix = nullptr;
    magmaDouble_ptr eigenvectors = nullptr;
    magmaDouble_ptr rank_u_update = nullptr;
    for (size_t i=0; i < POPULATION_SIZE; i++) magma_dmalloc(&changes[i], dimension);
    magma_dmalloc(&eigenvalues, dimension);
    magma_dmalloc(&zs, dimension);
    magma_dmalloc(&temporary_x, dimension);
    magma_dmalloc(&inverse_root_times_change_in_mean, dimension);
    magma_dmalloc(&max_coordinates, dimension);
    magma_dmalloc(&eigenvalues_matrix, rounded_dimension * dimension);
    magma_dmalloc(&eigenvectors, rounded_dimension * dimension);
    magma_dmalloc(&rank_u_update, rounded_dimension * dimension);
    magmablas_dlaset(MagmaFull, dimension, 1, MAX_SCORE, MAX_SCORE, max_coordinates, dimension, queue);
    magmablas_dlaset(MagmaFull, dimension, dimension, 0, 0, eigenvalues_matrix, rounded_dimension, queue);
    const magma_int_t nb = magma_get_dsytrd_nb(dimension);
    const magma_int_t lwork = magma_dmake_lwork(std::max(2 * dimension + dimension * nb, 1 + 6 * dimension + 2 * dimension * dimension));
    const magma_int_t liwork = 3 + 5 * dimension;
    magmaDouble_ptr work = nullptr;
    magma_int_t *iwork = nullptr;
    magma_dmalloc_cpu(&work, lwork);
    magma_imalloc_cpu(&iwork, liwork);
    std::unique_ptr<magma_int_t> temp_storage = std::make_unique<magma_int_t>();
    std::unique_ptr<magma_int_t> info = std::make_unique<magma_int_t>();
    magmaDouble_ptr local_vector = nullptr;
    magmaDouble_ptr means_cpu = nullptr;
    magmaDouble_ptr local_matrix = nullptr;
    magma_dmalloc_cpu(&local_vector, dimension);
    magma_dmalloc_cpu(& means_cpu, dimension);
    magma_dmalloc_cpu(&local_matrix, dimension * dimension);
    if (initial_variables) {
        step_size = 1;
        params = (std::array<float, Variables::num_parameters>)*initial_variables;
        for (size_t i=0; i < dimension; i++) means_cpu[i] = params[i];
    } else {
        for (size_t i = 0; i < 6 * WEIGHT_PARAMETER_COUNT; i++) means_cpu[i] = INITIAL_MEAN;
#ifdef OPTIMIZE_RATINGS
        for (size_t i = 6 * WEIGHT_PARAMETER_COUNT; i < 6 * WEIGHT_PARAMETER_COUNT + NUM_CARDS; i++) means_cpu[i] = 1.f;
        constexpr size_t start_index = 6 * WEIGHT_PARAMETER_COUNT + NUM_CARDS;
#else
        constexpr size_t start_index = 6 * WEIGHT_PARAMETER_COUNT;
#endif
        for (size_t i = start_index; i < dimension; i++) means_cpu[i] = INITIAL_MEAN;
    }
    magma_dsetvector(dimension, means_cpu, 1, mean, 1, queue);
    for (size_t i=0; i < dimension * dimension; i++) local_matrix[i] = 0;
    for (size_t generation=0; generation < num_generations; generation++) {
        auto start = std::chrono::high_resolution_clock::now();
        if (generation > 0) {
            magma_dcopymatrix(dimension, dimension, covariance, rounded_dimension, eigenvectors, rounded_dimension, queue);
            magma_dsyevd_gpu(MagmaVec, MagmaLower, dimension, eigenvectors, rounded_dimension, local_vector,
                             local_matrix, dimension, work, lwork, iwork, liwork, info.get());
//            magma_dsyevdx_m(magma_num_gpus(), MagmaVec, MagmaRangeAll, MagmaLower, dimension, local_matrix,
//                            dimension, 0.0, 0.0, 0, 0, temp_storage.get(), local_vector, work, lwork,
//                            iwork, liwork, info.get());
            magma_dsetmatrix(dimension, dimension, local_matrix, dimension, eigenvectors, rounded_dimension, queue);
            std::fill(local_matrix, local_matrix + dimension * dimension, 0);
            for (size_t i = 0; i < dimension; i++) {
                local_matrix[i + i * dimension] = std::sqrt(local_vector[i]);
                inverse_eigenvalues[i] = 1 / local_matrix[i + i * dimension];
            }
            magma_dsetmatrix(dimension, dimension, local_matrix, dimension, eigenvalues_matrix, rounded_dimension,
                             queue);
            magma_dsymm(MagmaRight, MagmaLower, dimension, dimension, 1, eigenvalues_matrix, rounded_dimension,
                        eigenvectors, rounded_dimension, 0, rank_u_update, rounded_dimension, queue);
            auto finish_solver = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> solver_diff = finish_solver - start;
            std::cout << "Took " << solver_diff.count() << " seconds to solve for eigenvalues and eigenvectors with status "
                      << *info << std::endl;
        }
        for (size_t i=0; i < POPULATION_SIZE; i++) {
            for (size_t j=0; j < Variables::num_parameters; j++) local_vector[j] = unit_normal(gen);
            if (generation > 0) {
                magma_dsetvector(dimension, local_vector, 1, zs, 1, queue);
                magma_dgemv(MagmaNoTrans, dimension, dimension, 1, rank_u_update, rounded_dimension, zs, 1,
                            0, changes[i], 1, queue);
                magma_dgetvector(dimension, changes[i], 1, local_vector, 1, queue);
            }
            for (size_t j=0; j < dimension; j++) {
                local_vector[j] = std::max(std::min(local_vector[j], (10 - means_cpu[j]) / step_size), -means_cpu[j] / step_size);
                params[j] = local_vector[j] * step_size + means_cpu[j];
            }
            magma_dsetvector(dimension, local_vector, 1, changes[i], 1, queue);
            population[i] = Variables(params);
        }
        for (size_t i = 0; i < PICKS_PER_GENERATION; i++) chosen_picks[i] = picks[pick_selector(gen)];
        losses = run_simulations(population, chosen_picks, temperature, constants);
        for (size_t i=0; i < POPULATION_SIZE; i++) {
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
        auto output_result = [=](const std::array<double, 4> &arr) {
            std::cout << "Loss: " << std::setw(WIDTH) << arr[0]
                      << " Categorical Cross-Entropy Loss: " << std::setw(WIDTH) << arr[1]
                      << " Negative Log Accuracy Loss: " << std::setw(WIDTH) << arr[2]
                      << " Accuracy Metric: " << std::setw(WIDTH) << arr[3]
                      << std::endl;
        };
        std::cout << "Generation " << generation << std::endl;
        std::ostringstream out_file_name;
        std::ostringstream std_devs_file_name;
        std::ostringstream mean_file_name;
        out_file_name << "output/variables-" << generation << ".json";
        std_devs_file_name << "output/std-devs-" << generation << ".json";
        mean_file_name << "output/means-" << generation << ".json";
        save_variables(population[indexed_accuracies[0].first], out_file_name.str());
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
        if (POPULATION_SIZE % 2 == 0) {
            std::array<double, 4> med_low = losses[indexed_accuracies[POPULATION_SIZE / 2 - 1].first];
            std::array<double, 4> med_hi = losses[indexed_accuracies[POPULATION_SIZE / 2].first];
            output_result({(med_low[0] + med_hi[0]) / 2, (med_low[1] + med_hi[1]) / 2,
                           (med_low[2] + med_hi[2]) / 2, (med_low[3] + med_hi[3]) / 2});
        } else {
            output_result(losses[indexed_accuracies[POPULATION_SIZE / 2].first]);
        }
        std::cout << "Worst Loss:     ";
        output_result(losses[indexed_accuracies[POPULATION_SIZE - 1].first]);
        magmablas_dlaset(MagmaFull, dimension, 1, 0, 0, change_in_mean, dimension, queue);
        for (size_t i=0; i < KEEP_BEST; i++) magma_daxpy(dimension, KEEP_WEIGHTS[i], changes[indexed_losses[i].first],
                                                         1, change_in_mean, 1, queue);
        magma_dgetvector(dimension, mean, 1, local_vector, 1, queue);
        for (size_t i=0; i < dimension; i++) params[i] = local_vector[i];
        save_variables(Variables(params), "previous_mean.json");
        magma_dgetvector(dimension, change_in_mean, 1, local_vector, 1, queue);
        for (size_t i=0; i < dimension; i++) params[i] = local_vector[i];
        save_variables(Variables(params), "change_in_mean.json");
        magma_daxpy(dimension, step_size * LEARNING_RATE, change_in_mean, 1, mean, 1, queue);
        std::fill(local_vector, local_vector + dimension, 0);
        magma_dgetvector(dimension, mean, 1, local_vector, 1, queue);
        for (size_t i=0; i < dimension; i++) params[i] = local_vector[i];
        save_variables(Variables(params), "output/new_mean.json");
        if (generation > 0) {
            for (size_t i = 0; i < dimension; i++) local_matrix[i + i * i] = inverse_eigenvalues[i];
            magma_dsetmatrix(dimension, dimension, local_matrix, dimension, eigenvalues_matrix, rounded_dimension,
                             queue);
            magma_dgemv(MagmaTrans, dimension, dimension, 1, eigenvectors, rounded_dimension, change_in_mean,
                        1, 0, inverse_root_times_change_in_mean, 1, queue);
            magma_dsymv(MagmaLower, dimension, 1, eigenvalues_matrix, rounded_dimension,  inverse_root_times_change_in_mean,
                        1, 0, temporary_x, 1, queue);
            magma_dgemv(MagmaNoTrans, dimension, dimension, 1, eigenvectors, rounded_dimension, temporary_x,
                        1, 0, inverse_root_times_change_in_mean, 1, queue);
        } else {
            magma_dcopyvector(dimension, change_in_mean, 1, inverse_root_times_change_in_mean, 1, queue);
        }
        magma_dscal(dimension, 1 - iso_discount_factor, iso_evolution_path, 1, queue);
        magma_daxpy(dimension, std::sqrt(iso_discount_factor * (2 - iso_discount_factor) * variance_selection_mass),
                    inverse_root_times_change_in_mean, 1, iso_evolution_path, 1, queue);
        double cs = 0;
        magma_dscal(dimension, 1 - aniso_discount_factor, aniso_evolution_path, 1, queue);
        if (magma_dnrm2(dimension, iso_evolution_path, 1, queue) / std::sqrt(1 - std::pow((1 - iso_discount_factor), 2*(generation + 1)))
            <= (1.4f + 2.f / (Variables::num_parameters + 1)) * expected_unit_norm) {
            magma_daxpy(dimension, std::sqrt(aniso_discount_factor * (2 - aniso_discount_factor) * variance_selection_mass),
                        change_in_mean, 1, aniso_evolution_path, 1, queue);
        } else {
            cs = rank_one_learning_rate * aniso_discount_factor * (2 - aniso_discount_factor);
        }
        double negative_weight = Variables::num_parameters / magma_ddot(dimension, inverse_root_times_change_in_mean,
                                                                       1, inverse_root_times_change_in_mean, 1, queue);
        for (size_t i=0; i < KEEP_BEST; i++) normalized_weights[i] = KEEP_WEIGHTS[i];
        for (size_t i=KEEP_BEST; i < POPULATION_SIZE; i++) normalized_weights[i] = negative_weight * KEEP_WEIGHTS[i];
        magmablas_dlaset(MagmaFull, dimension, dimension, 0, 0, rank_u_update, rounded_dimension, queue);
        for (size_t i=0; i < POPULATION_SIZE; i++) magma_dsyr(MagmaLower, dimension, normalized_weights[i],
                                                              changes[indexed_losses[i].first], 1, rank_u_update,
                                                              rounded_dimension, queue);
        magmablas_dgeadd2(dimension, dimension, rank_u_learning_rate, rank_u_update, rounded_dimension,
                          (1 - rank_one_learning_rate - rank_u_learning_rate + cs), covariance, rounded_dimension, queue);
        magma_dsyr(MagmaLower, dimension, rank_one_learning_rate, aniso_evolution_path, 1, covariance,
                   rounded_dimension, queue);
        step_size = step_size * (float)std::exp(iso_discount_factor / iso_damping_factor
                                                  * (magma_dnrm2(dimension, iso_evolution_path, 1, queue) / expected_unit_norm - 1));
        std::cout << "New step size " << step_size << std::endl;
        magma_dgetmatrix(dimension, dimension, covariance, rounded_dimension, local_matrix, dimension, queue);
        magma_dgetvector(dimension, mean, 1, means_cpu, 1, queue);
        if (generation % 25 == 24) {
            std::ofstream covariance_file("output/latest_covariance.bin", std::ios::out | std::ios::binary);
            covariance_file.write(reinterpret_cast<char *>(local_matrix), sizeof(double) * dimension * dimension);
        }
        for (size_t i=0; i < dimension; i++) params[i] = means_cpu[i];
        save_variables(Variables(params), mean_file_name.str());
        for (size_t i=0; i < dimension; i++) {
            params[i] = std::sqrt(local_matrix[i + i * dimension]) * step_size ;
            if (i % 153 == 0) std::cout << local_matrix[i + i * dimension] << ", ";
        }
        std::cout << std::endl;
        save_variables(Variables(params), std_devs_file_name.str());
        const auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        std::cout << "Generation took " << diff.count() << " seconds" << std::endl << std::endl;
    }
    magma_dgetvector(dimension, mean, 1, local_vector, 1, queue);
    for (size_t i=0; i < Variables::num_parameters; i++) params[i] = local_vector[i];
    return Variables(params);
}