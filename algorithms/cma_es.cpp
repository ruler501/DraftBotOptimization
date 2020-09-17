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
#include <random>
#include <sstream>
#include <vector>

#include <cuda_runtime_api.h>
#include <cusolverDn.h>

#include "../draftbot_optimization.h"
#include "shared/matrix_types.h"
#include "shared/parameters.h"

Variables optimize_variables(const float temperature, const std::vector<Pick>& picks, const size_t num_generations,
                             const std::shared_ptr<const Constants>& constants,
                             const std::shared_ptr<const Variables>& initial_variables, const size_t seed) {
    std::cout << "Beginning optimize_variables with population size of " << POPULATION_SIZE << " and " << picks.size()
              << " picks, sampling " << PICKS_PER_GENERATION << " per generation" << std::endl << std::endl;
    std::mt19937_64 gen{seed};
    std::normal_distribution<float> unit_normal(0, 1);
    std::uniform_int_distribution<size_t> pick_selector(0, picks.size() - 1);
    cusolverDnHandle_t cusolver_handle;
    cusolverStatus_t  cusolver_status = cusolverDnCreate(&cusolver_handle);
#ifdef DEBUG_LOG
    std::cout << __LINE__ << ": " << cusolver_status << std::endl;
#endif
    cublasHandle_t cublas_handle;
    cublasStatus_t cublas_status = cublasCreate_v2(&cublas_handle);
    cublasSetAtomicsMode(cublas_handle, CUBLAS_ATOMICS_ALLOWED);
#ifdef DEBUG_LOG
    std::cout << __LINE__ << ": " << cublas_status << std::endl;
#endif
    DeviceSquareMatrix<float, Variables::num_parameters> gpu_matrix(cusolver_handle, cublas_handle);
    DeviceVector<float, Variables::num_parameters> mean(cusolver_handle, cublas_handle);
    DeviceVector<float, Variables::num_parameters> iso_evolution_path(cusolver_handle, cublas_handle);
    DeviceVector<float, Variables::num_parameters> aniso_evolution_path(cusolver_handle, cublas_handle);
    gpu_matrix.set(0, 1, __LINE__);
    iso_evolution_path.set(0, (size_t)__LINE__);
    aniso_evolution_path.set(0, (size_t)__LINE__);
    float step_size = INITIAL_STD_DEV;
    const std::array<float, POPULATION_SIZE> PRELIMINARY_WEIGHTS = [] {
        std::array<float, POPULATION_SIZE> result{};
        for (size_t i = 0; i < POPULATION_SIZE; i++) result[i] = std::log(KEEP_BEST + 0.5f) - std::log((float)i + 1);
        return result;
    }();
    const float sum_preliminary_weights_positive = std::accumulate(PRELIMINARY_WEIGHTS.begin(),
                                                                   PRELIMINARY_WEIGHTS.begin() + KEEP_BEST, 0.f);
    const float variance_selection_mass = std::pow(sum_preliminary_weights_positive, 2.f)
                                           / std::accumulate(PRELIMINARY_WEIGHTS.begin(),
                                                             PRELIMINARY_WEIGHTS.begin() + KEEP_BEST,
                                                            0.f, [](float w1, float w2){ return w1 + w2 * w2; });
    const float sum_preliminary_weights_negative = std::accumulate(PRELIMINARY_WEIGHTS.begin() + KEEP_BEST,
                                                                   PRELIMINARY_WEIGHTS.end(), 0.f);
    const float negative_selection_mass = std::pow(sum_preliminary_weights_negative, 2.f)
                                            / std::accumulate(PRELIMINARY_WEIGHTS.begin() + KEEP_BEST,
                                                              PRELIMINARY_WEIGHTS.end(), 0.f,
                                                              [](float w1, float w2){ return w1 + w2 * w2; });
    const float rank_one_learning_rate = ALPHA_COVARIANCE / (std::pow(Variables::num_parameters + 1.3f, 2.f) + variance_selection_mass);
    const float rank_mu_learning_rate = std::min(1.f - rank_one_learning_rate,
                                                  ALPHA_COVARIANCE * (variance_selection_mass - 2
                                                                             + 1 / variance_selection_mass)
                                                  / (std::pow(Variables::num_parameters + 2.f, 2.f)
                                                        + ALPHA_COVARIANCE * variance_selection_mass / 2));
    const float iso_discount_factor = 1.f / 500.f;
    // const float iso_discount_factor = (variance_selection_mass + 2) / (Variables::num_parameters + variance_selection_mass + 5);
    const float iso_damping_factor = 1 + 2 * std::max(0.f, std::sqrt((variance_selection_mass - 1) / (Variables::num_parameters + 1)) - 1)
                                        + iso_discount_factor;
    const float aniso_discount_factor = 1.f / 700.f;
    // const float aniso_discount_factor = (4 + variance_selection_mass / Variables::num_parameters)
    //                                         / (Variables::num_parameters + 4 + 2 * variance_selection_mass / Variables::num_parameters);
    const float alpha_mu = 1 + rank_one_learning_rate / rank_mu_learning_rate;
    const float alpha_variance_mass = 1 + 2 * negative_selection_mass / (variance_selection_mass + 2);
    const float alpha_pos_def = (1 - rank_one_learning_rate - rank_mu_learning_rate)
                                / Variables::num_parameters / rank_mu_learning_rate;
    const float negative_scaling = std::min(std::min(alpha_mu, alpha_variance_mass), alpha_pos_def) / std::abs(sum_preliminary_weights_negative);
    const std::array<float, POPULATION_SIZE> KEEP_WEIGHTS = [&] {
        std::array<float, POPULATION_SIZE> result{};
        for (size_t i = 0; i < KEEP_BEST; i++) result[i] = PRELIMINARY_WEIGHTS[i] / sum_preliminary_weights_positive;
        for (size_t i = KEEP_BEST; i < POPULATION_SIZE; i++) result[i] =  PRELIMINARY_WEIGHTS[i] * negative_scaling;
        return result;
    }();
    const float sum_keep_weights = std::accumulate(KEEP_WEIGHTS.begin(), KEEP_WEIGHTS.end(), 0.f);
    const float expected_unit_norm = (float)std::sqrt(Variables::num_parameters)
                                     * (1 - 1 / 4.f / Variables::num_parameters
                                        + 1 / 21.f / Variables::num_parameters / Variables::num_parameters);
    const float covariance_factor = 1 - rank_one_learning_rate - rank_mu_learning_rate * sum_keep_weights;
    std::cout << "variance_selection_mass: " << variance_selection_mass << ", rank_one_learning_rate: "
              << rank_one_learning_rate << ", rank_mu_learning_rate: " << rank_mu_learning_rate << ", iso_discount_factor: "
              << iso_discount_factor << ", iso_damping_factor: " << iso_damping_factor << ", aniso_discount_factor: "
              << aniso_discount_factor << ", alpha_mu: " << alpha_mu << ", alpha_variance_mass: " << alpha_variance_mass
              << ", alpha_pos_def: " << alpha_pos_def << ", sum_keep_weights: " << sum_keep_weights << ", expected_unit_norm: "
              << expected_unit_norm << ", covariance_factor: " << covariance_factor << std::endl;
//    std::cout << "Sum of KEEP_WEIGHTS: " << sum_keep_weights << std::endl;
//    std::cout << "variance_selection_mass: " << variance_selection_mass << std::endl;
//    std::cout << "KEEP_WEIGHTS: ";
//    for (size_t i=0; i < POPULATION_SIZE; i++) std::cout << KEEP_WEIGHTS[i] << ", ";
//    std::cout << std::endl;
    std::vector<Variables> population(POPULATION_SIZE);
    std::vector<Pick> chosen_picks(PICKS_PER_GENERATION);
    std::array<std::array<double, 4>, POPULATION_SIZE> losses{};
    std::array<std::pair<size_t, double>, POPULATION_SIZE> indexed_losses{};
    std::array<std::pair<size_t, double>, POPULATION_SIZE> indexed_accuracies{};
    std::array<float, Variables::num_parameters> params{};
    std::array<float, POPULATION_SIZE> normalized_weights{};
    std::array<float, Variables::num_parameters> sqrt_eigenvalues{};
    std::array<DeviceVector<float, Variables::num_parameters>, POPULATION_SIZE> changes{};
    DeviceVector<float, Variables::num_parameters> temp_vec_a(cusolver_handle, cublas_handle, __LINE__);
    DeviceVector<float, Variables::num_parameters> change_in_mean(cusolver_handle, cublas_handle, __LINE__);
    HostVector<float, Variables::num_parameters> local_vector(cusolver_handle, cublas_handle, __LINE__);
    HostVector<float, Variables::num_parameters> local_vector2(cusolver_handle, cublas_handle, __LINE__);
    HostVector<float, Variables::num_parameters> local_vector3(cusolver_handle, cublas_handle, __LINE__);
    HostVector<float, Variables::num_parameters> mean_cpu(cusolver_handle, cublas_handle, __LINE__);
    HostSquareMatrix<float, Variables::num_parameters> local_matrix(gpu_matrix, __LINE__);
    if (initial_variables) {
        step_size = 1;
        params = (std::array<float, Variables::num_parameters>)*initial_variables;
        for (size_t i=0; i < Variables::num_parameters; i++) mean_cpu[i] = params[i];
    } else {
        mean_cpu.set(INITIAL_MEAN, (size_t)__LINE__);
    }
    mean.copy(mean_cpu, __LINE__);
    for (size_t generation=0; generation < num_generations; generation++) {
        auto start = std::chrono::high_resolution_clock::now();
        if (generation > 0) {
            temp_vec_a = gpu_matrix.symmetric_eigen_decomposition(__LINE__);
            local_vector.copy(temp_vec_a, __LINE__);
            for (size_t i=0; i < Variables::num_parameters; i++) params[i] = local_vector[i];
            save_variables(Variables(params), "eigenvalues.json");
            for (size_t i = 0; i < Variables::num_parameters; i++) sqrt_eigenvalues[i] = std::sqrt(local_vector[i]);
            auto finish_solver = std::chrono::high_resolution_clock::now();
            std::chrono::duration<float> solver_diff = finish_solver - start;
            std::cout << "Took " << solver_diff.count() << " seconds to solve for eigenvalues and eigenvectors.\n";
        }
        for (size_t i=0; i < POPULATION_SIZE; i++) {
            if (generation > 0) {
                for (size_t j=0; j < Variables::num_parameters; j++) local_vector[j] = sqrt_eigenvalues[j] * unit_normal(gen);
                temp_vec_a.copy(local_vector, __LINE__ * 100 + i);
                changes[i] = gpu_matrix * temp_vec_a;
                local_vector.copy(changes[i], __LINE__ * 100 + i);
            } else {
                for (size_t j=0; j < Variables::num_parameters; j++) local_vector[j] = unit_normal(gen);
            }
            for (size_t j=0; j < Variables::num_parameters; j++) {
                local_vector[j] = std::max(std::min(local_vector[j], (MAX_SCORE - mean_cpu[j]) / step_size), -mean_cpu[j] / step_size);
                params[j] = local_vector[j] * step_size + mean_cpu[j];
            }
            changes[i].copy(local_vector, __LINE__ * 100 + i);
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
        save_variables(population[indexed_losses[0].first], out_file_name.str());
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
        if constexpr (POPULATION_SIZE % 2 == 0) {
            std::array<double, 4> med_low = losses[indexed_accuracies[POPULATION_SIZE / 2 - 1].first];
            std::array<double, 4> med_hi = losses[indexed_accuracies[POPULATION_SIZE / 2].first];
            output_result({(med_low[0] + med_hi[0]) / 2, (med_low[1] + med_hi[1]) / 2,
                           (med_low[2] + med_hi[2]) / 2, (med_low[3] + med_hi[3]) / 2});
        } else {
            output_result(losses[indexed_accuracies[POPULATION_SIZE / 2].first]);
        }
        std::cout << "Worst Loss:     ";
        output_result(losses[indexed_accuracies[POPULATION_SIZE - 1].first]);
        change_in_mean.set(0, (size_t)__LINE__);
        for (size_t i=0; i < KEEP_BEST; i++) change_in_mean += KEEP_WEIGHTS[i] * changes[indexed_losses[i].first];
        mean += step_size * LEARNING_RATE * change_in_mean;
        if (generation > 0) {
            gpu_matrix.multiply_by_vector(true, 1, change_in_mean, 0, temp_vec_a, __LINE__);
            local_vector.copy(temp_vec_a, __LINE__);
            for (size_t i = 0; i < Variables::num_parameters; i++) local_vector[i] /= sqrt_eigenvalues[i];
            temp_vec_a.copy(local_vector, __LINE__);
            temp_vec_a = gpu_matrix * temp_vec_a;
        } else {
            temp_vec_a.copy(change_in_mean);
        }
        iso_evolution_path *= 1 - iso_discount_factor;
        iso_evolution_path += std::sqrt(iso_discount_factor * (2 - iso_discount_factor) * variance_selection_mass) * temp_vec_a;
        local_vector.copy(iso_evolution_path, __LINE__);
        for (size_t i=0; i < Variables::num_parameters; i++) params[i] = local_vector[i];
        save_variables(Variables(params), "iso_evolution_path.json");
        const float iso_norm = local_vector.norm();
        float cs = 0;
        aniso_evolution_path *= 1 - aniso_discount_factor;
        if (iso_norm / std::sqrt(1 - std::pow((1 - iso_discount_factor), 2*(generation + 2)))
            <= (1.4f + 2.f / (Variables::num_parameters + 1)) * expected_unit_norm) {
            aniso_evolution_path += std::sqrt(aniso_discount_factor * (2 - aniso_discount_factor) * variance_selection_mass) * change_in_mean;
        } else {
            cs = rank_one_learning_rate * aniso_discount_factor * (2 - aniso_discount_factor);
        }
        local_vector.copy(aniso_evolution_path, __LINE__);
        for (size_t i=0; i < Variables::num_parameters; i++) params[i] = local_vector[i];
        save_variables(Variables(params), "aniso_evolution_path.json");
        for (size_t i=0; i < KEEP_BEST; i++) normalized_weights[i] = KEEP_WEIGHTS[i];
        if (generation > 0) {
            for (size_t i=KEEP_BEST; i < POPULATION_SIZE; i++) {
                gpu_matrix.multiply_by_vector(true, 1, changes[indexed_losses[i].first], 0, temp_vec_a, __LINE__ * 100 + i);
                local_vector.copy(temp_vec_a, __LINE__ * 100 + i);
                for (size_t j = 0; j < Variables::num_parameters; j++) local_vector[j] /= sqrt_eigenvalues[j];
                temp_vec_a.copy(local_vector, __LINE__ * 100 + i);
                temp_vec_a = gpu_matrix * temp_vec_a;
                local_vector.copy(temp_vec_a, __LINE__ * 100 + i);
                float norm_squared = local_vector.norm_squared();
                float multiplier = Variables::num_parameters / norm_squared;
                normalized_weights[i] = KEEP_WEIGHTS[i] * multiplier;
            }
        } else {
            for (size_t i=KEEP_BEST; i < POPULATION_SIZE; i++) {
                local_vector.copy(changes[indexed_losses[i].first], __LINE__ * 100 + i);
                float multiplier = Variables::num_parameters / local_vector.norm_squared();
                normalized_weights[i] = KEEP_WEIGHTS[i] * multiplier;
            }
        }
        gpu_matrix.copy(local_matrix, __LINE__);
        gpu_matrix *= covariance_factor + cs;
        gpu_matrix.symmetric_rank_one_update(rank_one_learning_rate, aniso_evolution_path, __LINE__);
        for (size_t i=0; i < POPULATION_SIZE; i++) gpu_matrix.symmetric_rank_one_update(normalized_weights[i] * rank_mu_learning_rate,
                                                                                       changes[indexed_losses[i].first], __LINE__ * 100 + i);
        local_matrix.copy(gpu_matrix, __LINE__);
        step_size *= std::exp((iso_norm / expected_unit_norm - 1) * iso_discount_factor);
        std::cout << "New step size " << step_size << ", norm_squared: " << iso_norm << std::endl;
        mean_cpu.copy(mean, __LINE__);
        if (generation % 25 == 24) {
            std::ofstream covariance_file("output/latest_covariance.bin", std::ios::out | std::ios::binary);
            covariance_file.write(reinterpret_cast<const char *>(local_matrix.begin()), sizeof(float) * local_matrix.size());
        }
        for (size_t i=0; i < Variables::num_parameters; i++) params[i] = mean_cpu[i];
        save_variables(Variables(params), mean_file_name.str());
        for (size_t i=0; i < Variables::num_parameters; i++) params[i] = local_matrix(i, i);
        save_variables(Variables(params), std_devs_file_name.str());
        const auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> diff = end - start;
        std::cout << "Generation took " << diff.count() << " seconds" << std::endl << std::endl;
    }
    for (size_t i=0; i < Variables::num_parameters; i++) params[i] = mean_cpu[i];
    return Variables(params);
}