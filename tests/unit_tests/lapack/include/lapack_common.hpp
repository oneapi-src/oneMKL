/*******************************************************************************
* Copyright 2021 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
* http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing,
* software distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions
* and limitations under the License.
*
*
* SPDX-License-Identifier: Apache-2.0
*******************************************************************************/

#ifndef _LAPACK_COMMON_HPP__
#define _LAPACK_COMMON_HPP__

#include <CL/sycl.hpp>
#include <chrono>
#include <complex>
#include <iostream>
#include <random>
#include <sstream>
#include <thread>
#include <type_traits>

#include "oneapi/mkl/types.hpp"
#include "test_helper.hpp"

namespace global {
extern std::stringstream log;
extern std::array<char, 1024> buffer;
extern std::string pad;
} // namespace global

template <typename T>
struct complex_info {
    using real_type = T;
    static const bool is_complex = false;
};

template <typename T>
struct complex_info<std::complex<T>> {
    using real_type = T;
    static const bool is_complex = true;
};

template <typename fp>
fp get_real(fp val) {
    return val;
}
template <typename fp>
std::complex<fp> get_real(std::complex<fp> val) {
    return val.real();
}

template <typename fp>
fp get_conj(fp val) {
    return val;
}
template <typename fp>
std::complex<fp> get_conj(std::complex<fp> val) {
    return std::conj(val);
}

template <typename fp>
fp rand_scalar(uint64_t& seed) {
    std::minstd_rand rng(seed);
    seed = rng();
    return 2 * (static_cast<fp>(seed) / static_cast<fp>(rng.max())) - 0.0;
}
template <>
inline std::complex<float> rand_scalar(uint64_t& seed) {
    return std::complex<float>(rand_scalar<float>(seed), rand_scalar<float>(seed));
}
template <>
inline std::complex<double> rand_scalar(uint64_t& seed) {
    return std::complex<double>(rand_scalar<double>(seed), rand_scalar<double>(seed));
}

template <typename vec>
void rand_matrix(uint64_t& seed, oneapi::mkl::transpose trans, int64_t m, int64_t n, vec& M,
                 int64_t ld, int64_t offset = 0) {
    using fp = typename vec::value_type;
    if (trans == oneapi::mkl::transpose::nontrans)
        for (int64_t col = 0; col < n; col++)
            for (int64_t row = 0; row < m; row++)
                M[offset + row + col * ld] = rand_scalar<fp>(seed);
    else
        for (int64_t row = 0; row < m; row++)
            for (int64_t col = 0; col < n; col++)
                M[offset + col + row * ld] = rand_scalar<fp>(seed);
}

template <typename vec>
void rand_symmetric_matrix(uint64_t& seed, oneapi::mkl::uplo uplo, int64_t n, vec& M, int64_t ld,
                           int64_t offset = 0) {
    using fp = typename vec::value_type;
    using fp_real = typename complex_info<fp>::real_type;

    if (uplo == oneapi::mkl::uplo::upper)
        for (int64_t col = 0; col < n; col++)
            for (int64_t row = 0; row <= col; row++)
                M[offset + row + col * ld] = rand_scalar<fp>(seed);
    else
        for (int64_t col = 0; col < n; col++)
            for (int64_t row = col; row < n; row++)
                M[offset + row + col * ld] = rand_scalar<fp>(seed);
}

template <typename vec>
void rand_hermitian_matrix(uint64_t& seed, oneapi::mkl::uplo uplo, int64_t n, vec& M, int64_t ld,
                           int64_t offset = 0) {
    using fp = typename vec::value_type;
    using fp_real = typename complex_info<fp>::real_type;

    rand_symmetric_matrix(seed, uplo, n, M, ld, offset);
    for (int64_t diag = 0; diag < n; diag++)
        M[offset + diag + diag * ld] = rand_scalar<fp_real>(seed);
}

template <typename vec>
void rand_pos_def_matrix(uint64_t& seed, oneapi::mkl::uplo uplo, int64_t n, vec& M, int64_t ld,
                         int64_t offset = 0) {
    using fp = typename vec::value_type;
    using fp_real = typename complex_info<fp>::real_type;

    rand_hermitian_matrix(seed, uplo, n, M, ld, offset);
    for (int64_t diag = 0; diag < n; diag++)
        M[offset + diag + diag * ld] += static_cast<fp_real>(n);
    return;
}

template <typename T>
void symmetric_to_full(oneapi::mkl::uplo uplo, int64_t n, T& A, int64_t lda) {
    if (oneapi::mkl::uplo::upper == uplo)
        for (int64_t col = 0; col < n; col++)
            for (int64_t row = col + 1; row < n; row++)
                A[row + col * lda] = A[col + row * lda];
    else
        for (int64_t col = 0; col < n; col++)
            for (int64_t row = 0; row < col; row++)
                A[row + col * lda] = A[col + row * lda];
    return;
}

template <typename T>
void hermitian_to_full(oneapi::mkl::uplo uplo, int64_t n, T& A, int64_t lda) {
    for (int64_t diag = 0; diag < n; diag++)
        A[diag + diag * lda] = get_real(A[diag + diag * lda]);
    if (oneapi::mkl::uplo::upper == uplo)
        for (int64_t col = 0; col < n; col++)
            for (int64_t row = col + 1; row < n; row++)
                A[row + col * lda] = get_conj(A[col + row * lda]);
    else
        for (int64_t col = 0; col < n; col++)
            for (int64_t row = 0; row < col; row++)
                A[row + col * lda] = get_conj(A[col + row * lda]);
    return;
}

template <typename buffer_T>
struct is_buf {
    static constexpr bool value{ false };
};
template <typename T, int dimensions, typename AllocatorT>
struct is_buf<cl::sycl::buffer<T, dimensions, AllocatorT, void>> {
    static constexpr bool value{ true };
};

template <typename mem_T>
using is_buffer_type = typename std::enable_if<is_buf<mem_T>::value>::type*;
template <typename mem_T>
using is_not_buffer_type = typename std::enable_if<!is_buf<mem_T>::value>::type*;

template <typename mem_T, typename = void*>
struct mem_T_info {};
template <typename mem_T>
struct mem_T_info<mem_T, is_buffer_type<mem_T>> {
    using value_type = typename mem_T::value_type;
};
template <typename mem_T>
struct mem_T_info<mem_T, is_not_buffer_type<mem_T>> {
    using value_type = mem_T;
};

template <typename mem_T, typename T = typename mem_T::value_type, is_buffer_type<mem_T> = nullptr>
sycl::buffer<T, 1> device_alloc(sycl::queue queue, size_t count, size_t alignment = 4096) {
    sycl::buffer<T, 1> buf{ sycl::range<1>(count) };
    return buf;
}
template <typename mem_T, typename T = mem_T, is_not_buffer_type<mem_T> = nullptr>
T* device_alloc(sycl::queue queue, size_t count, size_t alignment = 4096) {
    const sycl::device dev = queue.get_device();
    const sycl::context ctx = queue.get_context();
    T* dev_ptr = (T*)oneapi::mkl::malloc_shared(alignment, count * sizeof(T), dev, ctx);
    return dev_ptr;
}

template <typename mem_T, is_buffer_type<mem_T> = nullptr>
void device_free(sycl::queue queue, mem_T buf) {}
template <typename mem_T, is_not_buffer_type<mem_T> = nullptr>
void device_free(sycl::queue queue, mem_T* dev_ptr) {
    const sycl::context ctx = queue.get_context();
    oneapi::mkl::free_shared(dev_ptr, ctx);
}

template <typename mem_T, is_buffer_type<mem_T> = nullptr>
void host_to_device_copy(sycl::queue queue, typename mem_T::value_type* source, mem_T dest,
                         size_t count) {
    queue.submit([&](sycl::handler& cgh) {
        auto dest_accessor =
            dest.template get_access<sycl::access::mode::discard_write>(cgh, sycl::range<1>(count));
        cgh.copy(source, dest_accessor);
    });
}
template <typename mem_T, is_not_buffer_type<mem_T> = nullptr>
sycl::event host_to_device_copy(sycl::queue queue, mem_T* source, mem_T* dest, size_t count) {
    return queue.memcpy(dest, source, count * sizeof(mem_T));
}

template <typename mem_T, is_buffer_type<mem_T> = nullptr>
void device_to_host_copy(sycl::queue queue, mem_T source, typename mem_T::value_type* dest,
                         size_t count) {
    queue.submit([&](sycl::handler& cgh) {
        auto source_accessor =
            source.template get_access<sycl::access::mode::read>(cgh, sycl::range<1>(count));
        cgh.copy(source_accessor, dest);
    });
}
template <typename mem_T, is_not_buffer_type<mem_T> = nullptr>
sycl::event device_to_host_copy(sycl::queue queue, mem_T* source, mem_T* dest, size_t count) {
    return queue.memcpy(dest, source, count * sizeof(mem_T));
}

enum class Dependency_Result { fail, pass, inconclusive, unknown };

sycl::event create_dependent_event(sycl::queue queue);
bool check_dependency(sycl::queue, sycl::event in_event, sycl::event func_event);
#endif // _LAPACK_COMMON_HPP__
