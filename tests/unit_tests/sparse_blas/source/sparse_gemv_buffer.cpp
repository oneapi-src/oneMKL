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

#include <complex>
#include <iostream>
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
         fpType beta, bool use_optimize) {
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

    auto ia_buf = make_buffer(ia_host);
    auto ja_buf = make_buffer(ja_host);
    auto a_buf = make_buffer(a_host);
    auto x_buf = make_buffer(x_host);
    auto y_buf = make_buffer(y_host);

    oneapi::mkl::sparse::matrix_handle_t handle = nullptr;
    sycl::event ev_release;
    try {
        CALL_RT_OR_CT(oneapi::mkl::sparse::init_matrix_handle, main_queue, &handle);

        CALL_RT_OR_CT(oneapi::mkl::sparse::set_csr_data, main_queue, handle, nrows, ncols, nnz,
                      index, ia_buf, ja_buf, a_buf);

        if (use_optimize) {
            CALL_RT_OR_CT(oneapi::mkl::sparse::optimize_gemv, main_queue, transpose_val, handle);
        }

        CALL_RT_OR_CT(oneapi::mkl::sparse::gemv, main_queue, transpose_val, alpha, handle, x_buf,
                      beta, y_buf);

        CALL_RT_OR_CT(ev_release = oneapi::mkl::sparse::release_matrix_handle, main_queue, &handle);
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
    auto y_acc = y_buf.template get_host_access(sycl::read_only);
    bool valid = check_equal_vector(y_acc, y_ref_host);

    ev_release.wait_and_throw();
    return static_cast<int>(valid);
}

class SparseGemvBufferTests : public ::testing::TestWithParam<sycl::device *> {};

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
    bool use_optimize = true;

    // Basic test
    EXPECT_TRUEORSKIP(test(dev, 4, 6, density_A_matrix, index_zero, transpose_val, fp_one, fp_zero,
                           use_optimize));
    // Test index_base 1
    EXPECT_TRUEORSKIP(test(dev, 4, 6, density_A_matrix, oneapi::mkl::index_base::one, transpose_val,
                           fp_one, fp_zero, use_optimize));
    // Test non-default alpha
    EXPECT_TRUEORSKIP(test(dev, 4, 6, density_A_matrix, index_zero, transpose_val,
                           set_fp_value<fpType>()(2.f, 1.5f), fp_zero, use_optimize));
    // Test non-default beta
    EXPECT_TRUEORSKIP(test(dev, 4, 6, density_A_matrix, index_zero, transpose_val, fp_one,
                           set_fp_value<fpType>()(3.2f, 1.f), use_optimize));
    // Test 0 alpha
    EXPECT_TRUEORSKIP(test(dev, 4, 6, density_A_matrix, index_zero, transpose_val, fp_zero, fp_one,
                           use_optimize));
    // Test 0 alpha and beta
    EXPECT_TRUEORSKIP(test(dev, 4, 6, density_A_matrix, index_zero, transpose_val, fp_zero, fp_zero,
                           use_optimize));
    // Test int64 indices
    EXPECT_TRUEORSKIP(test(dev, 27L, 13L, density_A_matrix, index_zero, transpose_val, fp_one,
                           fp_one, use_optimize));
    // Test without optimize_gemv
    EXPECT_TRUEORSKIP(
        test(dev, 4, 6, density_A_matrix, index_zero, transpose_val, fp_one, fp_zero, false));
}

TEST_P(SparseGemvBufferTests, RealSinglePrecision) {
    using fpType = float;
    test_helper<fpType>(GetParam(), oneapi::mkl::transpose::nontrans);
    test_helper<fpType>(GetParam(), oneapi::mkl::transpose::trans);
}

TEST_P(SparseGemvBufferTests, RealDoublePrecision) {
    using fpType = double;
    CHECK_DOUBLE_ON_DEVICE(GetParam());
    test_helper<fpType>(GetParam(), oneapi::mkl::transpose::nontrans);
    test_helper<fpType>(GetParam(), oneapi::mkl::transpose::trans);
}

TEST_P(SparseGemvBufferTests, ComplexSinglePrecision) {
    using fpType = std::complex<float>;
    test_helper<fpType>(GetParam(), oneapi::mkl::transpose::nontrans);
    test_helper<fpType>(GetParam(), oneapi::mkl::transpose::trans);
    test_helper<fpType>(GetParam(), oneapi::mkl::transpose::conjtrans);
}

TEST_P(SparseGemvBufferTests, ComplexDoublePrecision) {
    using fpType = std::complex<double>;
    CHECK_DOUBLE_ON_DEVICE(GetParam());
    test_helper<fpType>(GetParam(), oneapi::mkl::transpose::nontrans);
    test_helper<fpType>(GetParam(), oneapi::mkl::transpose::trans);
    test_helper<fpType>(GetParam(), oneapi::mkl::transpose::conjtrans);
}

INSTANTIATE_TEST_SUITE_P(SparseGemvBufferTestSuite, SparseGemvBufferTests,
                         testing::ValuesIn(devices), ::DeviceNamePrint());

} // anonymous namespace
