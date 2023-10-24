/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#ifndef _TEST_COMMON_HPP__
#define _TEST_COMMON_HPP__

#include <iostream>
#include <memory>
#include <limits>

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include "test_helper.hpp"

// Sparse BLAS domain needs to call more functions per test so we use this macro helper to select between runtime and compile dispatch for each function
#ifdef CALL_RT_API
#define CALL_RT_OR_CT(FUNC, QUEUE, ...) FUNC(QUEUE, __VA_ARGS__)
#else
#define CALL_RT_OR_CT(FUNC, QUEUE, ...) TEST_RUN_CT_SELECT(QUEUE, FUNC, __VA_ARGS__);
#endif

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

void print_error_code(sycl::exception const &e);

// Catch asynchronous exceptions.
struct exception_handler_t {
    void operator()(sycl::exception_list exceptions) {
        for (std::exception_ptr const &e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (sycl::exception const &e) {
                std::cout << "Caught asynchronous SYCL exception:\n" << e.what() << std::endl;
                print_error_code(e);
            }
        }
    }
};

// Use a unique_ptr to automatically free device memory on unique_ptr destruction.
template <class T>
auto malloc_device_uptr(sycl::queue q, std::size_t num_elts) {
    struct Deleter {
        sycl::queue q;
        Deleter(sycl::queue _q) : q(_q) {}
        void operator()(T *ptr) {
            sycl::free(ptr, q);
        }
    };
    return std::unique_ptr<T, Deleter>(sycl::malloc_device<T>(num_elts, q), Deleter(q));
}

// SYCL buffer creation helper.
template <typename vec>
sycl::buffer<typename vec::value_type, 1> make_buffer(const vec &v) {
    sycl::buffer<typename vec::value_type, 1> buf(v.data(), sycl::range<1>(v.size()));
    return buf;
}

template <typename fpType>
struct set_fp_value {
    inline fpType operator()(fpType real, fpType /*imag*/) {
        return real;
    }
};

template <typename scalarType>
struct set_fp_value<std::complex<scalarType>> {
    inline auto operator()(scalarType real, scalarType imag) {
        return std::complex<scalarType>(real, imag);
    }
};

template <typename fpType>
struct rand_scalar {
    inline fpType operator()(double min, double max) {
        return (fpType(std::rand()) / fpType(RAND_MAX)) * fpType(max - min) + fpType(min);
    }
};

template <typename fpType>
struct rand_scalar<std::complex<fpType>> {
    inline std::complex<fpType> operator()(double min, double max) {
        rand_scalar<fpType> rand;
        return std::complex<fpType>(rand(min, max), rand(min, max));
    }
};

template <typename fpType>
void rand_vector(std::vector<fpType> &v, std::size_t n) {
    using fpRealType = typename complex_info<fpType>::real_type;
    v.resize(n);
    rand_scalar<fpType> rand;
    for (std::size_t i = 0; i < n; i++) {
        v[i] = rand(fpRealType(-0.5), fpRealType(0.5));
    }
}

template <typename fpType>
void rand_matrix(std::vector<fpType> &m, oneapi::mkl::layout layout_val, std::size_t nrows,
                 std::size_t ncols, std::size_t ld) {
    using fpRealType = typename complex_info<fpType>::real_type;
    std::size_t outer_size = nrows;
    std::size_t inner_size = ncols;
    if (layout_val == oneapi::mkl::layout::col_major) {
        std::swap(outer_size, inner_size);
    }
    m.resize(outer_size * ld);
    rand_scalar<fpType> rand;
    for (std::size_t i = 0; i < outer_size; ++i) {
        std::size_t j = 0;
        for (; j < inner_size; ++j) {
            m[i * ld + j] = rand(fpRealType(-0.5), fpRealType(0.5));
        }
        for (; j < ld; ++j) {
            m[i * ld + j] = set_fp_value<fpType>()(-1.f, 0.f);
        }
    }
}

// Creating the 3arrays CSR representation (ia, ja, values)
// of general random sparse matrix
// with density (0 < density <= 1.0)
// -0.5 <= value < 0.5
// require_diagonal means all diagonal entries guaranteed to be nonzero
template <typename fpType, typename intType>
intType generate_random_matrix(const intType nrows, const intType ncols, const double density_val,
                               intType indexing, std::vector<intType> &ia, std::vector<intType> &ja,
                               std::vector<fpType> &a, bool require_diagonal = false) {
    intType nnz = 0;
    rand_scalar<double> rand_density;
    rand_scalar<fpType> rand_data;

    ia.push_back(indexing); // starting index of row0.
    for (intType i = 0; i < nrows; i++) {
        ia.push_back(nnz + indexing); // ending index of row_i.
        for (intType j = 0; j < ncols; j++) {
            const bool is_diag = require_diagonal && i == j;
            if (is_diag || (rand_density(0.0, 1.0) <= density_val)) {
                fpType val;
                if (is_diag) {
                    // Guarantee an amplitude >= 0.1
                    fpType sign = (std::rand() % 2) * 2 - 1;
                    val = rand_data(0.1, 0.5) * sign;
                }
                else {
                    val = rand_data(-0.5, 0.5);
                }
                a.push_back(val);
                ja.push_back(j + indexing);
                nnz++;
            }
        }
        ia[static_cast<std::size_t>(i) + 1] = nnz + indexing;
    }
    return nnz;
}

// Shuffle the 3arrays CSR representation (ia, ja, values)
// of any sparse matrix and set values serially from 0..nnz.
// Intended for use with sorting.
template <typename fpType, typename intType>
void shuffle_data(const intType *ia, intType *ja, fpType *a, const std::size_t nrows) {
    //
    // shuffle indices according to random seed
    //
    intType indexing = ia[0];
    for (std::size_t i = 0; i < nrows; ++i) {
        intType nnz_row = ia[i + 1] - ia[i];
        for (intType j = ia[i] - indexing; j < ia[i + 1] - indexing; ++j) {
            intType q = ia[i] - indexing + std::rand() % (nnz_row);
            // swap element i and q
            std::swap(ja[q], ja[j]);
            std::swap(a[q], a[j]);
        }
    }
}

inline void wait_and_free(sycl::queue &main_queue, oneapi::mkl::sparse::matrix_handle_t *p_handle) {
    main_queue.wait();
    sycl::event ev_release;
    CALL_RT_OR_CT(ev_release = oneapi::mkl::sparse::release_matrix_handle, main_queue, p_handle);
    ev_release.wait();
}

template <typename fpType>
bool check_equal(fpType x, fpType x_ref, double abs_error_margin, double rel_error_margin,
                 std::ostream &out) {
    using fpRealType = typename complex_info<fpType>::real_type;
    static_assert(std::is_floating_point_v<fpRealType>,
                  "Expected floating-point real or complex type.");

    const fpRealType epsilon = std::numeric_limits<fpRealType>::epsilon();
    const auto abs_bound = static_cast<fpRealType>(abs_error_margin) * epsilon;
    const auto rel_bound = static_cast<fpRealType>(rel_error_margin) * epsilon;

    const auto aerr = std::abs(x - x_ref);
    const auto rerr = aerr / std::abs(x_ref);
    const bool valid = (rerr <= rel_bound) || (aerr <= abs_bound);
    if (!valid) {
        out << "Mismatching results: actual = " << x << " vs. reference = " << x_ref << "\n";
        out << " relative error = " << rerr << " absolute error = " << aerr
            << " relative bound = " << rel_bound << " absolute bound = " << abs_bound << "\n";
    }
    return valid;
}

template <typename vecType1, typename vecType2>
bool check_equal_vector(const vecType1 &v, const vecType2 &v_ref, double abs_error_factor = 10.0,
                        double rel_error_factor = 200.0, std::ostream &out = std::cout) {
    using T = typename vecType2::value_type;
    std::size_t n = v.size();
    if (n != v_ref.size()) {
        out << "Mismatching size got " << n << " expected " << v_ref.size() << "\n";
        return false;
    }
    if (n == 0) {
        return true;
    }

    auto max_norm_ref =
        *std::max_element(std::begin(v_ref), std::end(v_ref),
                          [](const T &a, const T &b) { return std::abs(a) < std::abs(b); });
    // Heuristic for the average-case error margins
    double abs_error_margin =
        abs_error_factor * std::abs(max_norm_ref) * std::log2(static_cast<double>(n));
    double rel_error_margin = rel_error_factor * std::log2(static_cast<double>(n));

    constexpr int max_print = 20;
    int count = 0;
    bool valid = true;

    for (std::size_t i = 0; i < n; ++i) {
        // Allow to convert the unsigned index `i` to a signed one to keep this function generic and allow for `v` and `v_ref` to be a vector, a pointer or a random access iterator.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wsign-conversion"
        auto res = v[i];
        auto ref = v_ref[i];
#pragma clang diagnostic pop
        if (!check_equal(res, ref, abs_error_margin, rel_error_margin, out)) {
            out << " at index i =" << i << "\n";
            valid = false;
            ++count;
            if (count > max_print) {
                return valid;
            }
        }
    }

    return valid;
}

#endif // _TEST_COMMON_HPP__
