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
int test(sycl::device *dev, intType m, double density_A_matrix, oneapi::mkl::index_base index,
         oneapi::mkl::uplo uplo_val, oneapi::mkl::transpose transpose_val,
         oneapi::mkl::diag diag_val) {
    sycl::queue main_queue(*dev, exception_handler_t());

    intType int_index = (index == oneapi::mkl::index_base::zero) ? 0 : 1;
    const std::size_t mu = static_cast<std::size_t>(m);

    // Input matrix in CSR format
    std::vector<intType> ia_host, ja_host;
    std::vector<fpType> a_host;
    // Always require values to be present in the diagonal of the sparse matrix.
    // The values set in the matrix don't need to be 1s even if diag_val is unit.
    const bool require_diagonal = true;
    intType nnz = generate_random_matrix<fpType, intType>(
        m, m, density_A_matrix, int_index, ia_host, ja_host, a_host, require_diagonal);

    // Input dense vector.
    // The input `x` is initialized to random values on host and device.
    std::vector<fpType> x_host;
    rand_vector(x_host, mu);

    // Output and reference dense vectors.
    // They are both initialized with a dummy value to catch more errors.
    std::vector<fpType> y_host(mu, -2.0f);
    std::vector<fpType> y_ref_host(y_host);

    // Shuffle ordering of column indices/values to test sortedness
    shuffle_data(ia_host.data(), ja_host.data(), a_host.data(), mu);

    // TODO(Romain): Use buffer
    auto ia_buf = make_buffer(ia_host);
    auto ja_buf = make_buffer(ja_host);
    auto a_buf = make_buffer(a_host);
    auto x_buf = make_buffer(x_host);
    auto y_buf = make_buffer(y_host);

    sycl::event ev_release;
    oneapi::mkl::sparse::matrix_handle_t handle = nullptr;
    try {
        CALL_RT_OR_CT(oneapi::mkl::sparse::init_matrix_handle, main_queue, &handle);

        CALL_RT_OR_CT(oneapi::mkl::sparse::set_csr_data, main_queue, handle, m, m, nnz, index,
                      ia_buf, ja_buf, a_buf);

        CALL_RT_OR_CT(oneapi::mkl::sparse::optimize_trsv, main_queue, uplo_val, transpose_val,
                      diag_val, handle);

        CALL_RT_OR_CT(oneapi::mkl::sparse::trsv, main_queue, uplo_val, transpose_val, diag_val,
                      handle, x_buf, y_buf);

        CALL_RT_OR_CT(ev_release = oneapi::mkl::sparse::release_matrix_handle, main_queue, &handle);
    }
    catch (const sycl::exception &e) {
        std::cout << "Caught synchronous SYCL exception during sparse TRSV:\n"
                  << e.what() << std::endl;
        print_error_code(e);
        return 0;
    }
    catch (const oneapi::mkl::unimplemented &e) {
        wait_and_free(main_queue, &handle);
        return test_skipped;
    }
    catch (const std::runtime_error &error) {
        std::cout << "Error raised during execution of sparse TRSV:\n" << error.what() << std::endl;
        return 0;
    }

    // Compute reference.
    prepare_reference_trsv_data(ia_host.data(), ja_host.data(), a_host.data(), m, int_index,
                                uplo_val, transpose_val, diag_val, x_host.data(),
                                y_ref_host.data());

    // Compare the results of reference implementation and DPC++ implementation.
    auto y_acc = y_buf.template get_host_access(sycl::read_only);
    bool valid = check_equal_vector(y_acc, y_ref_host);

    ev_release.wait_and_throw();
    return static_cast<int>(valid);
}

class SparseTrsvBufferTests : public ::testing::TestWithParam<sycl::device *> {};

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
    oneapi::mkl::index_base index_zero = oneapi::mkl::index_base::zero;
    oneapi::mkl::uplo lower = oneapi::mkl::uplo::lower;
    oneapi::mkl::diag nonunit = oneapi::mkl::diag::nonunit;
    int m = 5;
    // Basic test
    EXPECT_TRUEORSKIP(
        test<fpType>(dev, m, density_A_matrix, index_zero, lower, transpose_val, nonunit));
    // Test index_base 1
    EXPECT_TRUEORSKIP(test<fpType>(dev, m, density_A_matrix, oneapi::mkl::index_base::one, lower,
                                   transpose_val, nonunit));
    // Test upper triangular matrix
    EXPECT_TRUEORSKIP(test<fpType>(dev, m, density_A_matrix, index_zero, oneapi::mkl::uplo::upper,
                                   transpose_val, nonunit));
    // Test unit diagonal matrix
    EXPECT_TRUEORSKIP(test<fpType>(dev, m, density_A_matrix, index_zero, lower, transpose_val,
                                   oneapi::mkl::diag::unit));
    // Test int64 indices
    EXPECT_TRUEORSKIP(
        test<fpType>(dev, 15L, density_A_matrix, index_zero, lower, transpose_val, nonunit));
}

TEST_P(SparseTrsvBufferTests, RealSinglePrecision) {
    using fpType = float;
    test_helper<fpType>(GetParam(), oneapi::mkl::transpose::nontrans);
    test_helper<fpType>(GetParam(), oneapi::mkl::transpose::trans);
}

TEST_P(SparseTrsvBufferTests, RealDoublePrecision) {
    using fpType = double;
    CHECK_DOUBLE_ON_DEVICE(GetParam());
    test_helper<fpType>(GetParam(), oneapi::mkl::transpose::nontrans);
    test_helper<fpType>(GetParam(), oneapi::mkl::transpose::trans);
}

TEST_P(SparseTrsvBufferTests, ComplexSinglePrecision) {
    using fpType = std::complex<float>;
    test_helper<fpType>(GetParam(), oneapi::mkl::transpose::nontrans);
    test_helper<fpType>(GetParam(), oneapi::mkl::transpose::trans);
    test_helper<fpType>(GetParam(), oneapi::mkl::transpose::conjtrans);
}

TEST_P(SparseTrsvBufferTests, ComplexDoublePrecision) {
    using fpType = std::complex<double>;
    CHECK_DOUBLE_ON_DEVICE(GetParam());
    test_helper<fpType>(GetParam(), oneapi::mkl::transpose::nontrans);
    test_helper<fpType>(GetParam(), oneapi::mkl::transpose::trans);
    test_helper<fpType>(GetParam(), oneapi::mkl::transpose::conjtrans);
}

INSTANTIATE_TEST_SUITE_P(SparseTrsvBufferTestSuite, SparseTrsvBufferTests,
                         testing::ValuesIn(devices), ::DeviceNamePrint());

} // anonymous namespace
