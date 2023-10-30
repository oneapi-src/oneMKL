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

#include <algorithm>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include "oneapi/mkl.hpp"
#include "oneapi/mkl/detail/config.hpp"
#include "sparse_reference.hpp"
#include "test_common.hpp"
#include "test_helper.hpp"

#include <gtest/gtest.h>

extern std::vector<sycl::device *> devices;

namespace {

template <typename fpType, typename intType>
int test(sycl::device *dev, intType nrows, intType ncols, double density_A_matrix,
         oneapi::mkl::index_base index, oneapi::mkl::transpose transpose_val, fpType alpha,
         fpType beta) {
    sycl::queue main_queue(*dev, exception_handler_t());

    intType int_index = (index == oneapi::mkl::index_base::zero) ? 0 : 1;
    std::size_t opa_nrows =
        static_cast<std::size_t>(transpose_val == oneapi::mkl::transpose::nontrans ? nrows : ncols);
    std::size_t opa_ncols =
        static_cast<std::size_t>(transpose_val == oneapi::mkl::transpose::nontrans ? ncols : nrows);

    // Input matrix in CSR format
    std::vector<intType> ia_host, ja_host;
    std::vector<fpType> a_host;
    intType nnz = generate_random_matrix<fpType, intType>(nrows, ncols, density_A_matrix, int_index,
                                                          ia_host, ja_host, a_host);

    // Input and output dense vectors
    // The input `x` and the input-output `y` are both initialized to random values on host and device.
    std::vector<fpType> x_host, y_host;
    rand_vector(x_host, opa_ncols);
    rand_vector(y_host, opa_nrows);
    std::vector<fpType> y_ref_host(y_host);

    // Shuffle ordering of column indices/values to test sortedness
    shuffle_data(ia_host.data(), ja_host.data(), a_host.data(), static_cast<std::size_t>(nrows));

    auto ia_usm_uptr = malloc_device_uptr<intType>(main_queue, ia_host.size());
    auto ja_usm_uptr = malloc_device_uptr<intType>(main_queue, ja_host.size());
    auto a_usm_uptr = malloc_device_uptr<fpType>(main_queue, a_host.size());
    auto x_usm_uptr = malloc_device_uptr<fpType>(main_queue, x_host.size());
    auto y_usm_uptr = malloc_device_uptr<fpType>(main_queue, y_host.size());

    intType *ia_usm = ia_usm_uptr.get();
    intType *ja_usm = ja_usm_uptr.get();
    fpType *a_usm = a_usm_uptr.get();
    fpType *x_usm = x_usm_uptr.get();
    fpType *y_usm = y_usm_uptr.get();

    std::vector<sycl::event> mat_dependencies;
    std::vector<sycl::event> gemv_dependencies;
    // Copy host to device
    mat_dependencies.push_back(
        main_queue.memcpy(ia_usm, ia_host.data(), ia_host.size() * sizeof(intType)));
    mat_dependencies.push_back(
        main_queue.memcpy(ja_usm, ja_host.data(), ja_host.size() * sizeof(intType)));
    mat_dependencies.push_back(
        main_queue.memcpy(a_usm, a_host.data(), a_host.size() * sizeof(fpType)));
    gemv_dependencies.push_back(
        main_queue.memcpy(x_usm, x_host.data(), x_host.size() * sizeof(fpType)));
    gemv_dependencies.push_back(
        main_queue.memcpy(y_usm, y_host.data(), y_host.size() * sizeof(fpType)));

    sycl::event ev_copy, ev_release;
    oneapi::mkl::sparse::matrix_handle_t handle = nullptr;
    try {
        sycl::event ev_set, ev_opt, ev_gemv;
        CALL_RT_OR_CT(oneapi::mkl::sparse::init_matrix_handle, main_queue, &handle);

        CALL_RT_OR_CT(ev_set = oneapi::mkl::sparse::set_csr_data, main_queue, handle, nrows, ncols,
                      nnz, index, ia_usm, ja_usm, a_usm, mat_dependencies);

        CALL_RT_OR_CT(ev_opt = oneapi::mkl::sparse::optimize_gemv, main_queue, transpose_val,
                      handle, { ev_set });

        gemv_dependencies.push_back(ev_opt);
        CALL_RT_OR_CT(ev_gemv = oneapi::mkl::sparse::gemv, main_queue, transpose_val, alpha, handle,
                      x_usm, beta, y_usm, gemv_dependencies);

        CALL_RT_OR_CT(ev_release = oneapi::mkl::sparse::release_matrix_handle, main_queue, &handle,
                      { ev_gemv });

        ev_copy = main_queue.memcpy(y_host.data(), y_usm, y_host.size() * sizeof(fpType), ev_gemv);
    }
    catch (const sycl::exception &e) {
        std::cout << "Caught synchronous SYCL exception during sparse GEMV:\n"
                  << e.what() << std::endl;
        print_error_code(e);
        return 0;
    }
    catch (const oneapi::mkl::unimplemented &e) {
        wait_and_free(main_queue, &handle);
        return test_skipped;
    }
    catch (const std::runtime_error &error) {
        std::cout << "Error raised during execution of sparse GEMV:\n" << error.what() << std::endl;
        return 0;
    }

    // Compute reference.
    prepare_reference_gemv_data(ia_host.data(), ja_host.data(), a_host.data(), nrows, ncols, nnz,
                                int_index, transpose_val, alpha, beta, x_host.data(),
                                y_ref_host.data());

    // Compare the results of reference implementation and DPC++ implementation.
    ev_copy.wait_and_throw();
    bool valid = check_equal_vector(y_host, y_ref_host);

    ev_release.wait_and_throw();
    return static_cast<int>(valid);
}

class SparseGemvUsmTests : public ::testing::TestWithParam<sycl::device *> {};

/**
 * Helper function to run tests in different configuration.
 *
 * @tparam fpType Complex or scalar, single or double precision type
 * @param dev Device to test
 * @param transpose_val Transpose value for the input matrix
 */
template <typename fpType>
void test_helper(sycl::device *dev, oneapi::mkl::transpose transpose_val) {
    double density_A_matrix = 0.8;
    fpType fp_zero = set_fp_value<fpType>()(0.f, 0.f);
    fpType fp_one = set_fp_value<fpType>()(1.f, 0.f);
    oneapi::mkl::index_base index_zero = oneapi::mkl::index_base::zero;
    // Basic test
    EXPECT_TRUEORSKIP(
        test(dev, 4, 6, density_A_matrix, index_zero, transpose_val, fp_one, fp_zero));
    // Test index_base 1
    EXPECT_TRUEORSKIP(test(dev, 4, 6, density_A_matrix, oneapi::mkl::index_base::one, transpose_val,
                           fp_one, fp_zero));
    // Test non-default alpha
    EXPECT_TRUEORSKIP(test(dev, 4, 6, density_A_matrix, index_zero, transpose_val,
                           set_fp_value<fpType>()(2.f, 1.5f), fp_zero));
    // Test non-default beta
    EXPECT_TRUEORSKIP(test(dev, 4, 6, density_A_matrix, index_zero, transpose_val, fp_one,
                           set_fp_value<fpType>()(3.2f, 1.f)));
    // Test 0 alpha
    EXPECT_TRUEORSKIP(
        test(dev, 4, 6, density_A_matrix, index_zero, transpose_val, fp_zero, fp_one));
    // Test 0 alpha and beta
    EXPECT_TRUEORSKIP(
        test(dev, 4, 6, density_A_matrix, index_zero, transpose_val, fp_zero, fp_zero));
    // Test int64 indices
    EXPECT_TRUEORSKIP(
        test(dev, 27L, 13L, density_A_matrix, index_zero, transpose_val, fp_one, fp_one));
}

TEST_P(SparseGemvUsmTests, RealSinglePrecision) {
    using fpType = float;
    test_helper<fpType>(GetParam(), oneapi::mkl::transpose::nontrans);
    test_helper<fpType>(GetParam(), oneapi::mkl::transpose::trans);
}

TEST_P(SparseGemvUsmTests, RealDoublePrecision) {
    using fpType = double;
    CHECK_DOUBLE_ON_DEVICE(GetParam());
    test_helper<fpType>(GetParam(), oneapi::mkl::transpose::nontrans);
    test_helper<fpType>(GetParam(), oneapi::mkl::transpose::trans);
}

TEST_P(SparseGemvUsmTests, ComplexSinglePrecision) {
    using fpType = std::complex<float>;
    test_helper<fpType>(GetParam(), oneapi::mkl::transpose::nontrans);
    test_helper<fpType>(GetParam(), oneapi::mkl::transpose::trans);
    test_helper<fpType>(GetParam(), oneapi::mkl::transpose::conjtrans);
}

TEST_P(SparseGemvUsmTests, ComplexDoublePrecision) {
    using fpType = std::complex<double>;
    CHECK_DOUBLE_ON_DEVICE(GetParam());
    test_helper<fpType>(GetParam(), oneapi::mkl::transpose::nontrans);
    test_helper<fpType>(GetParam(), oneapi::mkl::transpose::trans);
    test_helper<fpType>(GetParam(), oneapi::mkl::transpose::conjtrans);
}

INSTANTIATE_TEST_SUITE_P(SparseGemvUsmTestSuite, SparseGemvUsmTests, testing::ValuesIn(devices),
                         ::DeviceNamePrint());

} // anonymous namespace
