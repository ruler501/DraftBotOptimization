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


/***************************************************************************//**
    @return String describing cuBLAS errors (cublasStatus_t).
    CUDA provides cudaGetErrorString, but not cublasGetErrorString.

    @param[in]
    err     Error code.

    @ingroup magma_error_internal
*******************************************************************************/
extern "C"
const char* magma_cublasGetErrorStringCopy( cublasStatus_t err )
{
    switch( err ) {
        case CUBLAS_STATUS_SUCCESS:
            return "success";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "not initialized";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "out of memory";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "invalid value";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "architecture mismatch";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "memory mapping error";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "execution failed";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "internal error";

        default:
            return "unknown CUBLAS error code";
    }
}

void getvector(size_t dimension, magmaFloat_ptr source, magmaFloat_ptr dest, magma_queue_t queue, size_t line) {
    cublasStatus_t status;
    cudaError_t cuda_error;
    status = cublasGetVectorAsync(dimension, sizeof(float), source, 1, dest, 1, magma_queue_get_cuda_stream(queue));
    cuda_error = cudaStreamSynchronize(magma_queue_get_cuda_stream(queue));
//    std::cout << line << ": \"" << magma_cublasGetErrorStringCopy(status) << "\" \"" << cudaGetErrorString(cuda_error) << '"' << std::endl;
}

void setvector(size_t dimension, magmaFloat_ptr source, magmaFloat_ptr dest, magma_queue_t queue, size_t line) {
    cublasStatus_t status;
    cudaError_t cuda_error;
    status = cublasSetVectorAsync(dimension, sizeof(float), source, 1, dest, 1, magma_queue_get_cuda_stream(queue));
    cuda_error = cudaStreamSynchronize(magma_queue_get_cuda_stream(queue));
//    std::cout << line << ": \"" << magma_cublasGetErrorStringCopy(status) << "\" \"" << cudaGetErrorString(cuda_error) << '"' << std::endl;
}

void getmatrix(size_t dimension, magmaFloat_ptr source, size_t rounded_dimension, magmaFloat_ptr dest, magma_queue_t queue, size_t line) {
    cublasStatus_t status;
    cudaError_t cuda_error;
    status = cublasGetMatrixAsync(dimension, dimension, sizeof(float), source, rounded_dimension, dest, dimension, magma_queue_get_cuda_stream(queue));
    cuda_error = cudaStreamSynchronize(magma_queue_get_cuda_stream(queue));
//    std::cout << line << ": \"" << magma_cublasGetErrorStringCopy(status) << "\" \"" << cudaGetErrorString(cuda_error) << '"' << std::endl;
}

void setmatrix(size_t dimension, magmaFloat_ptr source, size_t rounded_dimension, magmaFloat_ptr dest, magma_queue_t queue, size_t line) {
    cublasStatus_t status;
    cudaError_t cuda_error;
    status = cublasSetMatrixAsync(dimension, dimension, sizeof(float), source, dimension, dest, rounded_dimension, magma_queue_get_cuda_stream(queue));
    cuda_error = cudaStreamSynchronize(magma_queue_get_cuda_stream(queue));
//    std::cout << line << ": \"" << magma_cublasGetErrorStringCopy(status) << "\" \"" << cudaGetErrorString(cuda_error) << '"' << std::endl;
}

Variables optimize_variables(const float temperature, const std::vector<Pick>& picks, const size_t num_generations,
                             const std::shared_ptr<const Constants>& constants,
                             const std::shared_ptr<const Variables>& initial_variables, const size_t seed) {
    std::cout << "Beginning optimize_variables with population size of " << POPULATION_SIZE << " and " << picks.size()
              << " picks, sampling " << PICKS_PER_GENERATION << " per generation" << std::endl << std::endl;
    std::mt19937_64 gen{seed};
    std::normal_distribution<float> unit_normal(0, 1);
    std::uniform_int_distribution<size_t> pick_selector(0, picks.size() - 1);
    cudaSetDeviceFlags(cudaDeviceBlockingSync);
    magma_int_t error;
    error = magma_init();
//    std::cout << __LINE__ << ": " << error << std::endl;
    magma_queue_t queue=nullptr;
//    magma_device_t devices[2];
//    magma_int_t device_count;
//    magma_getdevices(devices, 2, &device_count);
//    std::cout << device_count << " devices first of which has id " << devices[0] << std::endl;
    magma_queue_create(0, &queue);
    std::cout << "Device has " << magma_getdevice_multiprocessor_count() << " multiprocessors\n";
//    std::cout << "Queue: " << queue << std::endl;
    const magma_int_t rounded_dimension = magma_roundup(Variables::num_parameters, 32);
    const magma_int_t dimension = Variables::num_parameters;
    magmaFloat_ptr gpu_matrix = nullptr;
    magmaFloat_ptr mean = nullptr;
    magmaFloat_ptr iso_evolution_path = nullptr;
    magmaFloat_ptr aniso_evolution_path = nullptr;
    error = magma_smalloc(&gpu_matrix, rounded_dimension * dimension);
//    std::cout << __LINE__ << ": " << error << std::endl;
    error = magma_smalloc(&mean, dimension);
//    std::cout << __LINE__ << ": " << error << std::endl;
    error = magma_smalloc(&iso_evolution_path, dimension);
//    std::cout << __LINE__ << ": " << error << std::endl;
    error = magma_smalloc(&aniso_evolution_path, dimension);
//    std::cout << __LINE__ << ": " << error << std::endl;
    magmablas_slaset(MagmaFull, dimension, dimension, 0, 1, gpu_matrix, rounded_dimension, queue);
    magmablas_slaset(MagmaFull, dimension, 1, 0, 0, iso_evolution_path, dimension, queue);
    magmablas_slaset(MagmaFull, dimension, 1, 0, 0, aniso_evolution_path, dimension, queue);
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
    const float rank_mu_learning_rate = std::min(1.f - rank_one_learning_rate,
                                                  ALPHA_COVARIANCE * (variance_selection_mass - 2
                                                                             + 1 / variance_selection_mass)
                                                  / (std::pow(Variables::num_parameters + 2.f, 2.f)
                                                        + ALPHA_COVARIANCE * variance_selection_mass / 2));
    const float iso_discount_factor = (variance_selection_mass + 2) / (Variables::num_parameters + variance_selection_mass + 5);
    const float iso_damping_factor = 1 + 2 * std::max(0.f, std::sqrt((variance_selection_mass - 1) / (Variables::num_parameters + 1)) - 1)
                                        + iso_discount_factor;
    const float aniso_discount_factor = (4 + variance_selection_mass / Variables::num_parameters)
                                            / (Variables::num_parameters + 4 + 2 * variance_selection_mass / Variables::num_parameters);
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
    std::array<magmaFloat_ptr, POPULATION_SIZE> changes{};
    magmaFloat_ptr temp_vec_a = nullptr;
    magmaFloat_ptr change_in_mean = nullptr;
    magmaFloat_ptr temp_vec_b = nullptr;
    magmaFloat_ptr temp_matrix = nullptr;
    for (size_t i=0; i < POPULATION_SIZE; i++) {
        error = magma_smalloc(&changes[i], dimension);
//        std::cout << __LINE__ << "." << i << ": " << error << std::endl;
    }
    error = magma_smalloc(&temp_vec_a, dimension);
//    std::cout << __LINE__ << ": " << error << std::endl;
    error = magma_smalloc(&change_in_mean, dimension);
//    std::cout << __LINE__ << ": " << error << std::endl;
    error = magma_smalloc(&temp_vec_b, dimension);
//    std::cout << __LINE__ << ": " << error << std::endl;
    magma_smalloc(&temp_matrix, dimension * rounded_dimension);
    const magma_int_t nb = magma_get_dsytrd_nb(dimension);
    const magma_int_t lwork = std::max(2 * dimension + dimension * nb, 1 + 6 * dimension + 2 * dimension * dimension);
    const magma_int_t liwork = 3 + 5 * dimension;
    magmaFloat_ptr work = nullptr;
    magma_int_t *iwork = nullptr;
    magmaFloat_ptr local_vector = nullptr;
    magmaFloat_ptr local_vector2 = nullptr;
    magmaFloat_ptr local_vector3 = nullptr;
    magmaFloat_ptr mean_cpu = nullptr;
    magmaFloat_ptr local_matrix = nullptr;
    magmaFloat_ptr workspace_matrix = nullptr;
    error = magma_smalloc_cpu(&work, lwork);
//    std::cout << __LINE__ << ": " << error << std::endl;
    error = magma_imalloc_cpu(&iwork, liwork);
//    std::cout << __LINE__ << ": " << error << std::endl;
    error = magma_smalloc_cpu(&local_vector, dimension);
//    std::cout << __LINE__ << ": " << error << std::endl;
    error = magma_smalloc_cpu(&local_vector2, dimension);
//    std::cout << __LINE__ << ": " << error << std::endl;
    error = magma_smalloc_cpu(&local_vector3, dimension);
//    std::cout << __LINE__ << ": " << error << std::endl;
    error = magma_smalloc_cpu(&mean_cpu, dimension);
//    std::cout << __LINE__ << ": " << error << std::endl;
    error = magma_smalloc_cpu(&local_matrix, dimension * dimension);
//    std::cout << __LINE__ << ": " << error << std::endl;
    error = magma_smalloc_cpu(&workspace_matrix, dimension * dimension);
//    std::cout << __LINE__ << ": " << error << std::endl;
    magma_int_t info;
    if (initial_variables) {
        step_size = 1;
        params = (std::array<float, Variables::num_parameters>)*initial_variables;
        for (size_t i=0; i < dimension; i++) mean_cpu[i] = params[i];
    } else {
        std::fill(mean_cpu, mean_cpu + dimension, INITIAL_MEAN);
    }
    setvector(dimension, mean_cpu, mean, queue, __LINE__);
    getmatrix(dimension, gpu_matrix, rounded_dimension, local_matrix, queue, __LINE__);
    for (size_t generation=0; generation < num_generations; generation++) {
        auto start = std::chrono::high_resolution_clock::now();
        if (generation > 0) {
            magma_ssyevd_gpu(MagmaVec, MagmaLower, dimension, gpu_matrix, dimension, local_vector, workspace_matrix,
                             dimension, work, lwork, iwork, liwork, &info);
            for (size_t i=0; i < dimension; i++) params[i] = local_vector[i];
            save_variables(Variables(params), "eigenvalues.json");
            for (size_t i = 0; i < dimension; i++) sqrt_eigenvalues[i] = std::sqrt(local_vector[i]);
            auto finish_solver = std::chrono::high_resolution_clock::now();
            std::chrono::duration<float> solver_diff = finish_solver - start;
            std::cout << "Took " << solver_diff.count() << " seconds to solve for eigenvalues and eigenvectors with status "
                      << info << std::endl;
        }
        for (size_t i=0; i < POPULATION_SIZE; i++) {
            if (generation > 0) {
                for (size_t j=0; j < dimension; j++) local_vector[j] = sqrt_eigenvalues[j] * unit_normal(gen);
                setvector(dimension, local_vector, temp_vec_a, queue, __LINE__ * 100 + i);
                magma_sgemv(MagmaNoTrans, dimension, dimension, 1, gpu_matrix, rounded_dimension, temp_vec_a,
                            1, 0, changes[i], 1, queue);
                getvector(dimension, changes[i], local_vector, queue, __LINE__ * 100 + i);
            } else {
                for (size_t j=0; j < dimension; j++) local_vector[j] = unit_normal(gen);
            }
            for (size_t j=0; j < dimension; j++) {
                local_vector[j] = std::max(std::min(local_vector[j], (MAX_SCORE - mean_cpu[j]) / step_size), -mean_cpu[j] / step_size);
                params[j] = local_vector[j] * step_size + mean_cpu[j];
            }
            setvector(dimension, local_vector, changes[i], queue, __LINE__ * 100 + i);
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
        magmablas_slaset(MagmaFull, dimension, 1, 0, 0, change_in_mean, dimension, queue);
        for (size_t i=0; i < KEEP_BEST; i++) {
            magma_saxpy(dimension, KEEP_WEIGHTS[i], changes[indexed_losses[i].first],1, change_in_mean, 1, queue);
        }
        magma_saxpy(dimension, step_size * LEARNING_RATE, change_in_mean, 1, mean, 1, queue);
        if (generation > 0) {
            magma_sgemv(MagmaTrans, dimension, dimension, 1, gpu_matrix, rounded_dimension, change_in_mean,
                        1, 0, temp_vec_b, 1, queue);
            getvector(dimension, temp_vec_b, local_vector, queue, __LINE__);
            for (size_t i = 0; i < dimension; i++) local_vector[i] /= sqrt_eigenvalues[i];
            setvector(dimension, local_vector, temp_vec_b, queue, __LINE__);
            magma_sgemv(MagmaNoTrans, dimension, dimension, 1, gpu_matrix, rounded_dimension, temp_vec_b,
                        1, 0, temp_vec_a, 1, queue);
        } else {
            magma_scopyvector(dimension, change_in_mean, 1, temp_vec_a, 1, queue);
        }
        magma_sscal(dimension, 1 - iso_discount_factor, iso_evolution_path, 1, queue);
        magma_saxpy(dimension, std::sqrt(iso_discount_factor * (2 - iso_discount_factor) * variance_selection_mass),
                    temp_vec_a, 1, iso_evolution_path, 1, queue);
        getvector(dimension, iso_evolution_path, local_vector, queue, __LINE__);
        for (size_t i=0; i < dimension; i++) params[i] = local_vector[i];
        save_variables(Variables(params), "iso_evolution_path.json");
        const float iso_norm = std::sqrt(std::accumulate(local_vector, local_vector + dimension, 0.f,
                                                          [](float v1, float v2){ return v1 + v2 * v2; }));
        float cs = 0;
        magma_sscal(dimension, 1 - aniso_discount_factor, aniso_evolution_path, 1, queue);
        if (iso_norm / std::sqrt(1 - std::pow((1 - iso_discount_factor), 2*(generation + 2)))
            <= (1.4f + 2.f / (Variables::num_parameters + 1)) * expected_unit_norm) {
            magma_saxpy(dimension, std::sqrt(aniso_discount_factor * (2 - aniso_discount_factor) * variance_selection_mass),
                        change_in_mean, 1, aniso_evolution_path, 1, queue);
        } else {
            cs = rank_one_learning_rate * aniso_discount_factor * (2 - aniso_discount_factor);
        }
        getvector(dimension, aniso_evolution_path, local_vector, queue, __LINE__);
        for (size_t i=0; i < dimension; i++) params[i] = local_vector[i];
        save_variables(Variables(params), "aniso_evolution_path.json");
        for (size_t i=0; i < KEEP_BEST; i++) normalized_weights[i] = KEEP_WEIGHTS[i];
        for (size_t i=0; i < KEEP_BEST; i++) std::cout << i << ": " << rank_mu_learning_rate * normalized_weights[i] << std::endl;
        if (generation > 0) {
            for (size_t i=KEEP_BEST; i < POPULATION_SIZE; i++) {
                magma_sgemv(MagmaTrans, dimension, dimension, 1, gpu_matrix, rounded_dimension,
                            changes[indexed_losses[i].first], 1, 0, temp_vec_b, 1, queue);
                getvector(dimension, temp_vec_b, local_vector, queue, __LINE__);
                for (size_t j = 0; j < dimension; j++) local_vector[j] /= sqrt_eigenvalues[j];
                setvector(dimension, local_vector, temp_vec_b, queue, __LINE__);
                magma_sgemv(MagmaNoTrans, dimension, dimension, 1, gpu_matrix, rounded_dimension, temp_vec_b,
                            1, 0, temp_vec_a, 1, queue);
                getvector(dimension, temp_vec_a, local_vector, queue, __LINE__);
                float norm_squared = std::accumulate(local_vector, local_vector + dimension, 0.f,
                                                      [](float v1, float v2){ return v1 + v2 * v2; });
                float multiplier = Variables::num_parameters / norm_squared;
                normalized_weights[i] = KEEP_WEIGHTS[i] * multiplier;
            }
        } else {
            for (size_t i=KEEP_BEST; i < POPULATION_SIZE; i++) {
                getvector(dimension, changes[indexed_losses[i].first], local_vector, queue, __LINE__);
                float norm_squared = std::accumulate(local_vector, local_vector + dimension, 0.f,
                                                      [](float v1, float v2){ return v1 + v2 * v2; });
                float multiplier = Variables::num_parameters / norm_squared;
                normalized_weights[i] = KEEP_WEIGHTS[i] * multiplier;
            }
        }
        setmatrix(dimension, local_matrix, rounded_dimension, gpu_matrix, queue, __LINE__);
        std::cout << "Scaling matrix by " << 1 - rank_one_learning_rate - rank_mu_learning_rate + cs << std::endl;
        magmablas_slascl(MagmaLower, 0, 0, 1, 1 - rank_one_learning_rate - rank_mu_learning_rate + cs,
                         dimension, dimension, gpu_matrix, rounded_dimension, queue, &info);
        magma_scopymatrix(dimension, dimension, gpu_matrix, rounded_dimension, temp_matrix, rounded_dimension, queue);
        magma_spotri_gpu(MagmaLower, dimension, temp_matrix, rounded_dimension, &info);
        magma_ssymv(MagmaLower, dimension, 1, temp_matrix, rounded_dimension, aniso_evolution_path, 1, 0, temp_vec_a, 1, queue);
        getvector(dimension, aniso_evolution_path, local_vector2, queue, __LINE__);
        getvector(dimension, temp_vec_a, local_vector3, queue, __LINE__);
        float transformed_norm_aniso = 0;
        for (size_t i=0; i < dimension; i++) transformed_norm_aniso += local_vector2[i] * local_vector3[i];
        transformed_norm_aniso *= rank_one_learning_rate;
        if (transformed_norm_aniso <= -1) {
            std::cout << "Rank one update with aniso does not maintain positive definiteness got transformed norm: " << transformed_norm_aniso << std::endl;
            return {};
        }
        magma_ssyr(MagmaLower, dimension, rank_one_learning_rate, aniso_evolution_path, 1, gpu_matrix,
                   rounded_dimension, queue);
        for (size_t i=0; i < KEEP_BEST; i++) {
            magma_scopymatrix(dimension, dimension, gpu_matrix, rounded_dimension, temp_matrix, rounded_dimension, queue);
            magma_spotri_gpu(MagmaLower, dimension, temp_matrix, rounded_dimension, &info);
            magma_ssymv(MagmaLower, dimension, 1, temp_matrix, rounded_dimension, changes[indexed_losses[i].first], 1, 0, temp_vec_a, 1, queue);
            getvector(dimension, changes[indexed_losses[i].first], local_vector2, queue, __LINE__);
            getvector(dimension, temp_vec_a, local_vector3, queue, __LINE__);
            float transformed_norm_change = 0;
            for (size_t j=0; j < dimension; j++) transformed_norm_change += local_vector2[j] * local_vector3[j];
            transformed_norm_change *= rank_mu_learning_rate * normalized_weights[i];
            if (transformed_norm_change <= -1) {
                std::cout << "Rank one update with change " << i << " does not maintain positive definiteness got transformed norm: " << transformed_norm_change << std::endl;
                return {};
            }
            magma_ssyr(MagmaLower, dimension, normalized_weights[i] * rank_mu_learning_rate,
                       changes[indexed_losses[i].first], 1, gpu_matrix,
                       rounded_dimension, queue);
        }
        getmatrix(dimension, gpu_matrix, rounded_dimension, local_matrix, queue, __LINE__);
        step_size *= std::exp(iso_discount_factor / iso_damping_factor * (iso_norm / expected_unit_norm - 1));
        std::cout << "New step size " << step_size << ", norm_squared: " << iso_norm << std::endl;
        getvector(dimension, mean, mean_cpu, queue, __LINE__);
        if (generation % 25 == 24) {
            std::ofstream covariance_file("output/latest_covariance.bin", std::ios::out | std::ios::binary);
            covariance_file.write(reinterpret_cast<char *>(local_matrix), sizeof(float) * dimension * dimension);
        }
        for (size_t i=0; i < dimension; i++) params[i] = mean_cpu[i];
        save_variables(Variables(params), mean_file_name.str());
        for (size_t i=0; i < dimension; i++) {
            params[i] = local_matrix[i + i * dimension];
        }
        save_variables(Variables(params), std_devs_file_name.str());
        const auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> diff = end - start;
        std::cout << "Generation took " << diff.count() << " seconds" << std::endl << std::endl;
//        magma_free(temp_matrix);
    }
    magma_sgetvector(dimension, mean, 1, local_vector, 1, queue);
    for (size_t i=0; i < Variables::num_parameters; i++) params[i] = local_vector[i];
    return Variables(params);
}