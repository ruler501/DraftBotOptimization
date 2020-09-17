//
// Created by Devon Richards on 9/15/2020.
//

#ifndef DRAFTBOTOPTIMIZATION_MATRIX_TYPES_H
#define DRAFTBOTOPTIMIZATION_MATRIX_TYPES_H
#include <algorithm>
#include <iostream>

#include <cuda_runtime_api.h>
#include <cusolverDn.h>

template <typename Scalar, size_t columns, size_t rows, bool device, size_t padded_columns=columns>
class Matrix {
protected:
    Scalar* backing;
    cusolverDnHandle_t cusolver_handle{};
    cublasHandle_t cublas_handle{};

public:
    Matrix(size_t line=0) {
        if constexpr (device) {
            cudaError_t cuda_error = cudaMalloc((void **) &backing, sizeof(Scalar) * padded_columns * rows);
#ifdef DEBUG_LOG
            std::cout << line << ": \"" << cudaGetErrorString(cuda_error) << "\"\n";
#endif
        } else {
            backing = new float[padded_columns * rows];
#ifdef DEBUG_LOG
            std::cout << "Matrix: success\n";
#endif
        }
    }

    Matrix(cusolverDnHandle_t _cusolver_handler, cublasHandle_t _cublas_handle, size_t line=0) noexcept
            : cusolver_handle(_cusolver_handler), cublas_handle(_cublas_handle)
    {
        if constexpr (device) {
            cudaError_t cuda_error = cudaMalloc((void **) &backing, sizeof(Scalar) * padded_columns * rows);
#ifdef DEBUG_LOG
            std::cout << line << ": \"" << cudaGetErrorString(cuda_error) << "\"\n";
#endif
        } else {
            backing = new float[padded_columns * rows];
#ifdef DEBUG_LOG
            std::cout << "Matrix: success\n";
#endif
        }
    }

    void initialize_handles(cusolverDnHandle_t _cusolver_handler, cublasHandle_t _cublas_handle) {
        cusolver_handle = _cusolver_handler;
        cublas_handle = _cublas_handle;
    }

    template<bool other_device>
    explicit Matrix(const Matrix<Scalar, columns, rows, other_device, padded_columns>& other, size_t line=0) noexcept
            : Matrix(other.cusolver_handle, other.cublas_handle)  {
        copy(other, line);
    }

    template<bool other_device>
    Matrix& operator=(const Matrix<Scalar, columns, rows, other_device, padded_columns>& other) noexcept {
        copy(other);
        cusolver_handle = other.cusolver_handle;
        cublas_handle = other.cublas_handle;
        return *this;
    }

    Matrix(Matrix&& other) noexcept : backing(other.backing), cusolver_handle(other.cusolver_handle),
                                      cublas_handle(other.cublas_handle) {
        other.backing = nullptr;
    }

    Matrix& operator=(Matrix&& other) noexcept {
        if (backing != nullptr) {
            if constexpr (device) {
                cudaFree(backing);
            } else {
                delete[] backing;
            }
        }
        backing = other.backing;
        other.backing = nullptr;
        cusolver_handle = other.cusolver_handle;
        cublas_handle = other.cublas_handle;
        return *this;
    }

    ~Matrix() {
        if (backing != nullptr) {
            if constexpr (device) {
                cudaFree(backing);
            } else {
                delete[] backing;
            }
        }
    }

    Matrix& set(Scalar value, size_t line=0) {
        if constexpr (device) {
            Matrix<Scalar, columns, rows, false, padded_columns> temporary_storage(line);
            temporary_storage.set(value, line);
            copy(temporary_storage, line);
        } else {
            std::fill(backing, backing + padded_columns * rows, value);
        }
        return *this;
    }

    Matrix& set(Scalar off_diag_value, Scalar diag_value, size_t line=0) {
        if constexpr (device) {
            Matrix<Scalar, columns, rows, false, padded_columns> temporary_storage(line);
            temporary_storage.set(off_diag_value, diag_value, line);
            copy(temporary_storage, line);
        } else {
            std::fill(backing, backing + padded_columns * rows, off_diag_value);
            for (size_t i=0; i < std::min(rows, columns); i++) backing[i + padded_columns * i] = diag_value;
        }
        return *this;
    }

    template<bool other_device>
    void copy(const Matrix<Scalar, columns, rows, other_device, padded_columns>& other, const size_t line=0) {
        cudaMemcpyKind operation;
        if constexpr (device && other_device) {
            operation = cudaMemcpyDeviceToDevice;
        } else if constexpr(device && !other_device) {
            operation = cudaMemcpyHostToDevice;
        } else if constexpr (!device && other_device) {
            operation = cudaMemcpyDeviceToHost;
        } else {
            operation = cudaMemcpyHostToHost;
        }
        cudaDeviceSynchronize();
        cudaError_t cuda_error = cudaMemcpy(backing, other.backing, sizeof(Scalar) * padded_columns * rows, operation);
#ifdef DEBUG_LOG
        std::cout << line << ": \"" << cudaGetErrorString(cuda_error) << "\"\n";
#endif
    }

    Matrix<float, rows, 1, true>& multiply_by_vector(bool transpose, float alpha, const Matrix<float, rows, 1, true>& vec,
                                                        float beta, Matrix<float, columns, 1, true>& dest, size_t line=0) 
            requires device {
        cublasStatus_t cublas_status = cublasSgemv_v2(cublas_handle, transpose ? CUBLAS_OP_T : CUBLAS_OP_N, columns,
                                                      rows, &alpha, backing, padded_columns, vec.backing, 1, &beta,
                                                      dest.backing, 1);
#ifdef DEBUG_LOG
        std::cout << line << ": " << cublas_status << std::endl;
#endif
        return dest;
    }

    Matrix<float, rows, 1, true> operator*(const Matrix<float, rows, 1, true>& vec) requires device {
        Matrix<float, rows, 1, true> result(cusolver_handle, cublas_handle);
        multiply_by_vector(false, 1, vec, 0, result);
        return result;
    }

    Matrix& operator+=(const Matrix& other) requires(device && std::is_same_v<Scalar, float>) {
        float alpha = 1;
        cublasStatus_t cublas_status = cublasSaxpy_v2(cublas_handle, padded_columns * rows, &alpha, other.backing, 1, backing, 1);
#ifdef DEBUG_LOG
        std::cout << "Addition" << ": " << cublas_status << std::endl;
#endif
        return *this;
    }

    Matrix& operator*=(float alpha) requires(device && std::is_same_v<Scalar, float>)  {
        Matrix<float, columns, rows, false, padded_columns> temp;
        temp.copy(*this);
        for (size_t i = 0; i < columns; i++) {
            for (size_t j = 0; j < rows; j++) {
                temp(i, j) *= alpha;
            }
        }
        copy(temp);
//        cublasStatus_t cublas_status = cublasSscal_v2(cublas_handle, padded_columns * rows, &alpha, backing, 1);
#ifdef DEBUG_LOG
//        std::cout << "Scaling" << ": " << cublas_status << std::endl;
#endif
        return *this;
    }

    friend Matrix<Scalar, columns, rows, true, padded_columns> operator*(Scalar alpha, const Matrix& mat) {
        Matrix<Scalar, columns, rows, true, padded_columns> result;
        result.copy(mat);
        result *= alpha;
        return result;
    }

    Scalar& operator()(size_t column, size_t row) noexcept requires (!device) {
        return backing[row * padded_columns + column];
    }

    const Scalar& operator()(size_t column, size_t row) const noexcept requires (!device) {
        return backing[row * padded_columns + column];
    }

    Scalar& operator[](size_t index) noexcept requires (!device && rows == 1) {
        return (*this)(index, 0);
    }

    const Scalar& operator[](size_t index) const noexcept requires (!device && rows == 1) {
        return (*this)(index, 0);
    }

    Scalar& operator()(size_t index) noexcept requires (!device && rows == 1) {
        return (*this)(index, 0);
    }

    const Scalar& operator()(size_t index) const noexcept requires (!device && rows == 1) {
        return (*this)(index, 0);
    }

    const Scalar* begin() const noexcept requires(!device) {
        return backing;
    }

    const Scalar* end() const noexcept requires(!device) {
        return backing + padded_columns * rows;
    }

    Scalar norm_squared() const noexcept requires(!device && rows == 1) {
        return std::accumulate(begin(), end(), static_cast<Scalar>(0), [](Scalar v1, Scalar v2){ return v1 + v2 * v2; });
    }

    Scalar norm() const noexcept requires(!device && rows == 1) {
        return std::sqrt(norm_squared());
    }

    Matrix& symmetric_rank_one_update(Scalar alpha, const Matrix<float, columns, 1, true>& vec, size_t line=0) noexcept
            requires (device && rows == columns && std::is_same_v<Scalar, float>) {
        cublasStatus_t cublas_status = cublasSsyr(cublas_handle, CUBLAS_FILL_MODE_LOWER, columns, &alpha, vec.backing, 1,
                                                  backing, padded_columns);
#ifdef DEBUG_LOG
        std::cout << line << ": " << cublas_status << std::endl;
#endif
        return *this;
    }

    Matrix<Scalar, columns, 1, true> symmetric_eigen_decomposition(size_t line=0)
            requires (device && rows == columns && std::is_same_v<Scalar, float>) {
        Matrix<Scalar, columns, 1, true> eigenvalues(cusolver_handle, cublas_handle);
        int lwork;
        int* devInfo;
        cusolverStatus_t cusolver_status = cusolverDnSsyevd_bufferSize(cusolver_handle, CUSOLVER_EIG_MODE_VECTOR,
                                                                       CUBLAS_FILL_MODE_LOWER, columns, backing,
                                                                       padded_columns, eigenvalues.backing, &lwork);
#ifdef DEBUG_LOG
        std::cout << line << '.' << "1: " << cusolver_status << ", " << lwork << std::endl;
#endif
        float* workspace;
        cudaError_t cuda_error = cudaMalloc((void**)&workspace, sizeof(float) * lwork);
#ifdef DEBUG_LOG
        std::cout << line << ".2: \"" << cudaGetErrorString(cuda_error) << "\"\n";
#endif
        cuda_error = cudaMalloc((void**)&devInfo, sizeof(int));
#ifdef DEBUG_LOG
        std::cout << line << ".3: \"" << cudaGetErrorString(cuda_error) << "\"\n";
#endif
        cusolver_status = cusolverDnSsyevd(cusolver_handle, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, columns,
                                           backing, padded_columns, eigenvalues.backing, workspace, lwork, devInfo);
        cuda_error = cudaDeviceSynchronize();
#ifdef DEBUG_LOG
        std::cout << line << ".4: \"" << cudaGetErrorString(cuda_error) << "\" " << cusolver_status << std::endl;
#endif
        cudaFree(workspace);
        cudaFree(devInfo);
        return eigenvalues;
    }

    [[nodiscard]] size_t size() const noexcept {
        return rows * padded_columns;
    }

    friend Matrix<Scalar, rows, 1, !device, padded_columns>;
    friend Matrix<Scalar, rows, 1, device, padded_columns>;
    friend Matrix<Scalar, columns, rows, !device, padded_columns>;
    friend Matrix<Scalar, columns, columns, !device, padded_columns>;
    friend Matrix<Scalar, columns, columns, device, padded_columns>;
};

template <typename Scalar, size_t dimension, bool device>
using Vector = Matrix<Scalar, dimension, 1, device>;

template <typename Scalar, size_t dimension, bool device>
using SquareMatrix = Matrix<Scalar, dimension, dimension, device>;

template <typename Scalar, size_t dimension>
using DeviceSquareMatrix = SquareMatrix<Scalar, dimension, true>;

template <typename Scalar, size_t dimension>
using HostSquareMatrix = SquareMatrix<Scalar, dimension, false>;

template <typename Scalar, size_t dimension>
using DeviceVector = Vector<Scalar, dimension, true>;

template <typename Scalar, size_t dimension>
using HostVector = Vector<Scalar, dimension, false>;


#endif //DRAFTBOTOPTIMIZATION_MATRIX_TYPES_H
