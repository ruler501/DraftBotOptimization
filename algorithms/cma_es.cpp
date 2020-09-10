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

#include <Eigen/Dense>

#include "../draftbot_optimization.h"
#include "shared/parameters.h"

using Matrix = Eigen::MatrixXf;
using Vector = Eigen::VectorXf;

Variables optimize_variables(const float temperature, const std::vector<Pick>& picks, const size_t num_generations,
                             const std::shared_ptr<const Constants>& constants,
                             const std::shared_ptr<const Variables>& initial_variables, const size_t seed) {
    std::cout << "Beginning optimize_variables with population size of " << POPULATION_SIZE << " and " << picks.size()
              << " picks, sampling " << PICKS_PER_GENERATION << " per generation" << std::endl << std::endl;
    std::mt19937_64 gen{seed};
    std::normal_distribution<float> unit_normal(0, 1);
    std::uniform_int_distribution<size_t> pick_selector(0, picks.size() - 1);
    Matrix covariance = Matrix::Identity(Variables::num_parameters, Variables::num_parameters);
    Vector means  = INITIAL_MEAN * Vector::Ones(Variables::num_parameters);
    Vector iso_evolution_path = Vector::Zero(Variables::num_parameters);
    Vector aniso_evolution_path = Vector::Zero(Variables::num_parameters);
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
    Eigen::DiagonalMatrix<float, Variables::num_parameters> sqrt_eigenvalues;
    Matrix eigenvectors(Variables::num_parameters, Variables::num_parameters);
    Vector zs(Variables::num_parameters);
    std::array<Vector, POPULATION_SIZE> changes{Vector::Zero(Variables::num_parameters)};
    Vector temporary_x(Variables::num_parameters);
    Vector change_in_mean = Vector::Zero(Variables::num_parameters);
    Vector new_means(Variables::num_parameters);
    Vector inverse_root_times_change_in_mean(Variables::num_parameters);
    Matrix rank_u_update(Variables::num_parameters, Variables::num_parameters);
    Vector max_coordinates = MAX_SCORE * Vector::Ones(Variables::num_parameters);
    for (size_t i=0; i < POPULATION_SIZE; i++) changes[i] = Vector::Zero(Variables::num_parameters);
    for (size_t generation=0; generation < num_generations; generation++) {
        auto start = std::chrono::high_resolution_clock::now();
        Eigen::SelfAdjointEigenSolver<Matrix> solver(covariance);
        auto finish_solver = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> solver_diff = finish_solver - start;
        std::cout << "Took " << solver_diff.count() << " seconds to solve for eigenvalues and eigenvectors" << std::endl;
        eigenvectors = solver.eigenvectors();
        sqrt_eigenvalues = solver.eigenvalues().cwiseSqrt().asDiagonal();
        for (size_t i=0; i < POPULATION_SIZE; i++) {
            for (size_t j=0; j < Variables::num_parameters; j++) zs(j) = unit_normal(gen);
            changes[i] = eigenvectors * sqrt_eigenvalues * zs;
            changes[i] = changes[i].cwiseMin((max_coordinates - means) / step_size).cwiseMax(-means / step_size);
            temporary_x = means + step_size * changes[i];
            for (size_t j=0; j < Variables::num_parameters; j++) params[j] = temporary_x(j);
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
        std::ostringstream covariance_file_name;
        std::ostringstream mean_file_name;
        out_file_name << "output/variables-" << generation << ".json";
        covariance_file_name << "output/covariance-" << generation << ".json";
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
        change_in_mean = Vector::Zero(Variables::num_parameters);
        for (size_t i=0; i < KEEP_BEST; i++) {
            change_in_mean += KEEP_WEIGHTS[i] * changes[indexed_losses[i].first];
        }
        new_means = means + LEARNING_RATE * step_size * change_in_mean;
        inverse_root_times_change_in_mean = solver.operatorInverseSqrt() * change_in_mean;
        iso_evolution_path = (1 - iso_discount_factor) * iso_evolution_path
                + std::sqrt(iso_discount_factor * (2 - iso_discount_factor) * variance_selection_mass)
                    * inverse_root_times_change_in_mean;
        float cs = 0;
        if (iso_evolution_path.norm() / std::sqrt(1 - std::pow((1 - iso_discount_factor), 2*(generation + 1)))
            <= (1.4f + 2.f / (Variables::num_parameters + 1)) * expected_unit_norm) {
            aniso_evolution_path = (1 - aniso_discount_factor) * aniso_evolution_path
                   + std::sqrt(aniso_discount_factor * (2 - aniso_discount_factor) * variance_selection_mass)
                    * change_in_mean;
        } else {
            aniso_evolution_path = (1 - aniso_discount_factor) * aniso_evolution_path;
            cs = rank_one_learning_rate * aniso_discount_factor * (2 - aniso_discount_factor);
        }
        float negative_weight = Variables::num_parameters / inverse_root_times_change_in_mean.squaredNorm();
        for (size_t i=0; i < KEEP_BEST; i++) normalized_weights[i] = KEEP_WEIGHTS[i];
        for (size_t i=KEEP_BEST; i < POPULATION_SIZE; i++) normalized_weights[i] = negative_weight * KEEP_WEIGHTS[i];
        rank_u_update = Matrix::Zero(Variables::num_parameters, Variables::num_parameters);
        for (size_t i=0; i < POPULATION_SIZE; i++) {
            const Eigen::VectorXf& change = changes[indexed_losses[i].first];
            rank_u_update += normalized_weights[i] * change * change.transpose();
        }
        covariance = (1 - rank_one_learning_rate - rank_u_learning_rate * sum_keep_weights + cs) * covariance
                + rank_one_learning_rate * aniso_evolution_path * aniso_evolution_path.transpose()
                + rank_u_learning_rate * rank_u_update;
        step_size = step_size * (float)std::exp(iso_discount_factor / iso_damping_factor
                                                  * (iso_evolution_path.norm() / expected_unit_norm - 1));
        std::cout << "New step size " << step_size << std::endl;
        means = new_means;
#ifndef OPTIMIZE_RATINGS
        std::ofstream covariance_file(covariance_file_name.str());
        covariance_file << covariance;
#endif
        std::ofstream mean_file(mean_file_name.str());
        mean_file << means;
        const auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        std::cout << "Generation took " << diff.count() << " seconds" << std::endl << std::endl;
    }
    for (size_t i=0; i < Variables::num_parameters; i++) params[i] = means(i);
    return Variables(params);
}