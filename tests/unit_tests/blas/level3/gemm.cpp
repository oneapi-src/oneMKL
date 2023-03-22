/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif
#include "allocator_helper.hpp"
#include "cblas.h"
#include "oneapi/mkl.hpp"
#include "oneapi/mkl/detail/config.hpp"
#include "onemkl_blas_helper.hpp"
#include "reference_blas_templates.hpp"
#include "test_common.hpp"
#include "test_helper.hpp"

#include <gtest/gtest.h>

using namespace sycl;
using std::vector;

extern std::vector<sycl::device*> devices;

namespace {

template <typename Ta, typename Tc>
int test(device* dev, oneapi::mkl::layout layout, oneapi::mkl::transpose transa,
         oneapi::mkl::transpose transb, int m, int n, int k, int lda, int ldb, int ldc, Tc alpha,
         Tc beta) {
    // Prepare data.
    vector<Ta, allocator_helper<Ta, 64>> A, B;
    vector<Tc, allocator_helper<Tc, 64>> C, C_ref;

    rand_matrix(A, layout, transa, m, k, lda);
    rand_matrix(B, layout, transb, k, n, ldb);
    rand_matrix(C, layout, oneapi::mkl::transpose::nontrans, m, n, ldc);
    C_ref = C;

    // Call Reference GEMM.
    const int m_ref = m, n_ref = n, k_ref = k;
    const int lda_ref = lda, ldb_ref = ldb, ldc_ref = ldc;

    using Ta_ref = typename ref_type_info<Ta>::type;
    using Tc_ref = typename ref_type_info<Tc>::type;

    ::gemm(convert_to_cblas_layout(layout), convert_to_cblas_trans(transa),
           convert_to_cblas_trans(transb), &m_ref, &n_ref, &k_ref, (Tc_ref*)&alpha,
           (Ta_ref*)A.data(), &lda_ref, (Ta_ref*)B.data(), &ldb_ref, (Tc_ref*)&beta,
           (Tc_ref*)C_ref.data(), &ldc_ref);

    // Call DPC++ GEMM.

    // Catch asynchronous exceptions.
    auto exception_handler = [](exception_list exceptions) {
        for (std::exception_ptr const& e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (exception const& e) {
                std::cout << "Caught asynchronous SYCL exception during GEMM:\n"
                          << e.what() << std::endl;
                print_error_code(e);
            }
        }
    };

    queue main_queue(*dev, exception_handler);

    buffer<Ta, 1> A_buffer(A.data(), range<1>(A.size()));
    buffer<Ta, 1> B_buffer(B.data(), range<1>(B.size()));
    buffer<Tc, 1> C_buffer(C.data(), range<1>(C.size()));

    try {
#ifdef CALL_RT_API
        switch (layout) {
            case oneapi::mkl::layout::column_major:
                oneapi::mkl::blas::column_major::gemm(main_queue, transa, transb, m, n, k, alpha,
                                                      A_buffer, lda, B_buffer, ldb, beta, C_buffer,
                                                      ldc);
                break;
            case oneapi::mkl::layout::row_major:
                oneapi::mkl::blas::row_major::gemm(main_queue, transa, transb, m, n, k, alpha,
                                                   A_buffer, lda, B_buffer, ldb, beta, C_buffer,
                                                   ldc);
                break;
            default: break;
        }
#else
        switch (layout) {
            case oneapi::mkl::layout::column_major:
                TEST_RUN_CT_SELECT(main_queue, oneapi::mkl::blas::column_major::gemm, transa,
                                   transb, m, n, k, alpha, A_buffer, lda, B_buffer, ldb, beta,
                                   C_buffer, ldc);
                break;
            case oneapi::mkl::layout::row_major:
                TEST_RUN_CT_SELECT(main_queue, oneapi::mkl::blas::row_major::gemm, transa, transb,
                                   m, n, k, alpha, A_buffer, lda, B_buffer, ldb, beta, C_buffer,
                                   ldc);
                break;
            default: break;
        }
#endif
    }
    catch (exception const& e) {
        std::cout << "Caught synchronous SYCL exception during GEMM:\n" << e.what() << std::endl;
        print_error_code(e);
    }

    catch (const oneapi::mkl::unimplemented& e) {
        return test_skipped;
    }

    catch (const std::runtime_error& error) {
        std::cout << "Error raised during execution of GEMM:\n" << error.what() << std::endl;
    }

    // Compare the results of reference implementation and DPC++ implementation.
    auto C_accessor = C_buffer.template get_host_access(read_only);
    bool good = check_equal_matrix(C_accessor, C_ref, layout, m, n, ldc, 10 * k, std::cout);

    return (int)good;
}

class GemmTests : public ::testing::TestWithParam<std::tuple<sycl::device*, oneapi::mkl::layout>> {
};

TEST_P(GemmTests, Bfloat16Bfloat16FloatPrecision) {
    float alpha(2.0);
    float beta(3.0);
    EXPECT_TRUEORSKIP((test<oneapi::mkl::bfloat16, float>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::mkl::transpose::nontrans,
        oneapi::mkl::transpose::nontrans, 79, 83, 91, 103, 105, 106, alpha, beta)));
    EXPECT_TRUEORSKIP((test<oneapi::mkl::bfloat16, float>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::mkl::transpose::nontrans,
        oneapi::mkl::transpose::trans, 79, 83, 91, 103, 105, 106, alpha, beta)));
    EXPECT_TRUEORSKIP((test<oneapi::mkl::bfloat16, float>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::mkl::transpose::trans,
        oneapi::mkl::transpose::nontrans, 79, 83, 91, 103, 105, 106, alpha, beta)));
    EXPECT_TRUEORSKIP((test<oneapi::mkl::bfloat16, float>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::mkl::transpose::trans,
        oneapi::mkl::transpose::trans, 79, 83, 91, 103, 105, 106, alpha, beta)));
}

TEST_P(GemmTests, HalfHalfFloatPrecision) {
    float alpha(2.0);
    float beta(3.0);
    EXPECT_TRUEORSKIP((test<sycl::half, float>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::mkl::transpose::nontrans,
        oneapi::mkl::transpose::nontrans, 79, 83, 91, 103, 105, 106, alpha, beta)));
    EXPECT_TRUEORSKIP((test<sycl::half, float>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::mkl::transpose::nontrans,
        oneapi::mkl::transpose::trans, 79, 83, 91, 103, 105, 106, alpha, beta)));
    EXPECT_TRUEORSKIP((test<sycl::half, float>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::mkl::transpose::trans,
        oneapi::mkl::transpose::nontrans, 79, 83, 91, 103, 105, 106, alpha, beta)));
    EXPECT_TRUEORSKIP((test<sycl::half, float>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::mkl::transpose::trans,
        oneapi::mkl::transpose::trans, 79, 83, 91, 103, 105, 106, alpha, beta)));
}

TEST_P(GemmTests, RealHalfPrecision) {
    sycl::half alpha(2.0);
    sycl::half beta(3.0);
    EXPECT_TRUEORSKIP((test<sycl::half, sycl::half>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::mkl::transpose::nontrans,
        oneapi::mkl::transpose::nontrans, 79, 83, 91, 103, 105, 106, alpha, beta)));
    EXPECT_TRUEORSKIP((test<sycl::half, sycl::half>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::mkl::transpose::nontrans,
        oneapi::mkl::transpose::trans, 79, 83, 91, 103, 105, 106, alpha, beta)));
    EXPECT_TRUEORSKIP((test<sycl::half, sycl::half>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::mkl::transpose::trans,
        oneapi::mkl::transpose::nontrans, 79, 83, 91, 103, 105, 106, alpha, beta)));
    EXPECT_TRUEORSKIP((test<sycl::half, sycl::half>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::mkl::transpose::trans,
        oneapi::mkl::transpose::trans, 79, 83, 91, 103, 105, 106, alpha, beta)));
}

TEST_P(GemmTests, RealSinglePrecision) {
    float alpha(2.0);
    float beta(3.0);
    EXPECT_TRUEORSKIP((test<float, float>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::mkl::transpose::nontrans,
        oneapi::mkl::transpose::nontrans, 3, 8, 9, 103, 105, 106, alpha, beta)));
    EXPECT_TRUEORSKIP((test<float, float>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::mkl::transpose::nontrans,
        oneapi::mkl::transpose::nontrans, 79, 83, 91, 103, 105, 106, alpha, beta)));
    EXPECT_TRUEORSKIP((test<float, float>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::mkl::transpose::nontrans,
        oneapi::mkl::transpose::trans, 79, 83, 91, 103, 105, 106, alpha, beta)));
    EXPECT_TRUEORSKIP((test<float, float>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::mkl::transpose::trans,
        oneapi::mkl::transpose::nontrans, 79, 83, 91, 103, 105, 106, alpha, beta)));
    EXPECT_TRUEORSKIP((test<float, float>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::mkl::transpose::trans,
        oneapi::mkl::transpose::trans, 79, 83, 91, 103, 105, 106, alpha, beta)));
}

TEST_P(GemmTests, RealDoublePrecision) {
    double alpha(2.0);
    double beta(3.0);
    EXPECT_TRUEORSKIP((test<double, double>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::mkl::transpose::nontrans,
        oneapi::mkl::transpose::nontrans, 79, 83, 91, 103, 105, 106, alpha, beta)));
    EXPECT_TRUEORSKIP((test<double, double>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::mkl::transpose::nontrans,
        oneapi::mkl::transpose::trans, 79, 83, 91, 103, 105, 106, alpha, beta)));
    EXPECT_TRUEORSKIP((test<double, double>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::mkl::transpose::trans,
        oneapi::mkl::transpose::nontrans, 79, 83, 91, 103, 105, 106, alpha, beta)));
    EXPECT_TRUEORSKIP((test<double, double>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::mkl::transpose::trans,
        oneapi::mkl::transpose::trans, 79, 83, 91, 103, 105, 106, alpha, beta)));
}

TEST_P(GemmTests, ComplexSinglePrecision) {
    std::complex<float> alpha(2.0, -0.5);
    std::complex<float> beta(3.0, -1.5);
    EXPECT_TRUEORSKIP((test<std::complex<float>, std::complex<float>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::mkl::transpose::nontrans,
        oneapi::mkl::transpose::nontrans, 79, 83, 91, 103, 105, 106, alpha, beta)));
    EXPECT_TRUEORSKIP((test<std::complex<float>, std::complex<float>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::mkl::transpose::nontrans,
        oneapi::mkl::transpose::trans, 79, 83, 91, 103, 105, 106, alpha, beta)));
    EXPECT_TRUEORSKIP((test<std::complex<float>, std::complex<float>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::mkl::transpose::trans,
        oneapi::mkl::transpose::nontrans, 79, 83, 91, 103, 105, 106, alpha, beta)));
    EXPECT_TRUEORSKIP((test<std::complex<float>, std::complex<float>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::mkl::transpose::trans,
        oneapi::mkl::transpose::trans, 79, 83, 91, 103, 105, 106, alpha, beta)));
    EXPECT_TRUEORSKIP((test<std::complex<float>, std::complex<float>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::mkl::transpose::nontrans,
        oneapi::mkl::transpose::conjtrans, 79, 83, 91, 103, 105, 106, alpha, beta)));
    EXPECT_TRUEORSKIP((test<std::complex<float>, std::complex<float>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::mkl::transpose::trans,
        oneapi::mkl::transpose::conjtrans, 79, 83, 91, 103, 105, 106, alpha, beta)));
    EXPECT_TRUEORSKIP((test<std::complex<float>, std::complex<float>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::mkl::transpose::conjtrans,
        oneapi::mkl::transpose::nontrans, 79, 83, 91, 103, 105, 106, alpha, beta)));
    EXPECT_TRUEORSKIP((test<std::complex<float>, std::complex<float>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::mkl::transpose::conjtrans,
        oneapi::mkl::transpose::trans, 79, 83, 91, 103, 105, 106, alpha, beta)));
    EXPECT_TRUEORSKIP((test<std::complex<float>, std::complex<float>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::mkl::transpose::conjtrans,
        oneapi::mkl::transpose::conjtrans, 79, 83, 91, 103, 105, 106, alpha, beta)));
}

TEST_P(GemmTests, ComplexDoublePrecision) {
    std::complex<double> alpha(2.0, -0.5);
    std::complex<double> beta(3.0, -1.5);
    EXPECT_TRUEORSKIP((test<std::complex<double>, std::complex<double>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::mkl::transpose::nontrans,
        oneapi::mkl::transpose::nontrans, 79, 83, 91, 103, 105, 106, alpha, beta)));
    EXPECT_TRUEORSKIP((test<std::complex<double>, std::complex<double>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::mkl::transpose::nontrans,
        oneapi::mkl::transpose::trans, 79, 83, 91, 103, 105, 106, alpha, beta)));
    EXPECT_TRUEORSKIP((test<std::complex<double>, std::complex<double>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::mkl::transpose::trans,
        oneapi::mkl::transpose::nontrans, 79, 83, 91, 103, 105, 106, alpha, beta)));
    EXPECT_TRUEORSKIP((test<std::complex<double>, std::complex<double>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::mkl::transpose::trans,
        oneapi::mkl::transpose::trans, 79, 83, 91, 103, 105, 106, alpha, beta)));
    EXPECT_TRUEORSKIP((test<std::complex<double>, std::complex<double>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::mkl::transpose::nontrans,
        oneapi::mkl::transpose::conjtrans, 79, 83, 91, 103, 105, 106, alpha, beta)));
    EXPECT_TRUEORSKIP((test<std::complex<double>, std::complex<double>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::mkl::transpose::trans,
        oneapi::mkl::transpose::conjtrans, 79, 83, 91, 103, 105, 106, alpha, beta)));
    EXPECT_TRUEORSKIP((test<std::complex<double>, std::complex<double>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::mkl::transpose::conjtrans,
        oneapi::mkl::transpose::nontrans, 79, 83, 91, 103, 105, 106, alpha, beta)));
    EXPECT_TRUEORSKIP((test<std::complex<double>, std::complex<double>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::mkl::transpose::conjtrans,
        oneapi::mkl::transpose::trans, 79, 83, 91, 103, 105, 106, alpha, beta)));
    EXPECT_TRUEORSKIP((test<std::complex<double>, std::complex<double>>(
        std::get<0>(GetParam()), std::get<1>(GetParam()), oneapi::mkl::transpose::conjtrans,
        oneapi::mkl::transpose::conjtrans, 79, 83, 91, 103, 105, 106, alpha, beta)));
}

INSTANTIATE_TEST_SUITE_P(GemmTestSuite, GemmTests,
                         ::testing::Combine(testing::ValuesIn(devices),
                                            testing::Values(oneapi::mkl::layout::column_major,
                                                            oneapi::mkl::layout::row_major)),
                         ::LayoutDeviceNamePrint());

} // anonymous namespace
