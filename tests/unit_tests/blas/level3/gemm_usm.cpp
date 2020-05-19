/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include <CL/sycl.hpp>
#include "cblas.h"
#include "onemkl/detail/config.hpp"
#include "onemkl/onemkl.hpp"
#include "onemkl_blas_helper.hpp"
#include "reference_blas_templates.hpp"
#include "test_common.hpp"
#include "test_helper.hpp"
#include "usm_allocator_helper.hpp"

#include <gtest/gtest.h>

using namespace cl::sycl;
using std::vector;

extern std::vector<cl::sycl::device> devices;

namespace {

template <typename fp>
int test(const device& dev, onemkl::transpose transa, onemkl::transpose transb, int m, int n, int k,
         int lda, int ldb, int ldc, fp alpha, fp beta) {
    // Catch asynchronous exceptions.
    auto exception_handler = [](exception_list exceptions) {
        for (std::exception_ptr const& e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (exception const& e) {
                std::cout << "Caught asynchronous SYCL exception during GEMM:\n"
                          << e.what() << std::endl
                          << "OpenCL status: " << e.get_cl_code() << std::endl;
            }
        }
    };

    queue main_queue(dev, exception_handler);
    context cxt = main_queue.get_context();
    event done;
    std::vector<event> dependencies;

    // Prepare data.
    auto ua = usm_allocator_helper<fp, 64>(cxt, dev);
    vector<fp, decltype(ua)> A(ua), B(ua), C(ua), C_ref(ua);
    rand_matrix(A, transa, m, k, lda);
    rand_matrix(B, transb, k, n, ldb);
    rand_matrix(C, onemkl::transpose::nontrans, m, n, ldc);

    C_ref.resize(C.size());
    for (int i = 0; i < C.size(); i++)
        C_ref[i] = C[i];

    // Call Reference GEMM.
    const int m_ref = m, n_ref = n, k_ref = k;
    const int lda_ref = lda, ldb_ref = ldb, ldc_ref = ldc;

    using fp_ref = typename ref_type_info<fp>::type;

    ::gemm(convert_to_cblas_trans(transa), convert_to_cblas_trans(transb), &m_ref, &n_ref, &k_ref,
           (fp_ref*)&alpha, (fp_ref*)A.data(), &lda_ref, (fp_ref*)B.data(), &ldb_ref,
           (fp_ref*)&beta, (fp_ref*)C_ref.data(), &ldc_ref);

    // Call DPC++ GEMM.

    try {
#ifdef CALL_RT_API
        done = onemkl::blas::gemm(main_queue, transa, transb, m, n, k, alpha, A.data(), lda,
                                  B.data(), ldb, beta, C.data(), ldc, dependencies);
        done.wait();
#else
        TEST_RUN_CT(main_queue, onemkl::blas::gemm,
                    (main_queue, transa, transb, m, n, k, alpha, A.data(), lda, B.data(), ldb, beta,
                     C.data(), ldc, dependencies));
        main_queue.wait();
#endif
    }
    catch (exception const& e) {
        std::cout << "Caught synchronous SYCL exception during GEMM:\n"
                  << e.what() << std::endl
                  << "OpenCL status: " << e.get_cl_code() << std::endl;
    }

    catch (const backend_unsupported_exception& e) {
        return test_skipped;
    }

    catch (const std::runtime_error& error) {
        std::cout << "Error raised during execution of GEMM_USM:\n" << error.what() << std::endl;
    }

    // Compare the results of reference implementation and DPC++ implementation.

    bool good = check_equal_matrix(C, C_ref, m, n, ldc, 10 * k, std::cout);

    return (int)good;
}

class GemmUsmTests : public ::testing::TestWithParam<cl::sycl::device> {};

TEST_P(GemmUsmTests, RealSinglePrecision) {
    float alpha(2.0);
    float beta(3.0);
    EXPECT_TRUEORSKIP(test<float>(GetParam(), onemkl::transpose::nontrans,
                                  onemkl::transpose::nontrans, 79, 83, 91, 103, 105, 106, alpha,
                                  beta));
    EXPECT_TRUEORSKIP(test<float>(GetParam(), onemkl::transpose::nontrans, onemkl::transpose::trans,
                                  79, 83, 91, 103, 105, 106, alpha, beta));
    EXPECT_TRUEORSKIP(test<float>(GetParam(), onemkl::transpose::trans, onemkl::transpose::nontrans,
                                  79, 83, 91, 103, 105, 106, alpha, beta));
    EXPECT_TRUEORSKIP(test<float>(GetParam(), onemkl::transpose::trans, onemkl::transpose::trans,
                                  79, 83, 91, 103, 105, 106, alpha, beta));
}

TEST_P(GemmUsmTests, RealDoublePrecision) {
    double alpha(2.0);
    double beta(3.0);
    EXPECT_TRUEORSKIP(test<double>(GetParam(), onemkl::transpose::nontrans,
                                   onemkl::transpose::nontrans, 79, 83, 91, 103, 105, 106, alpha,
                                   beta));
    EXPECT_TRUEORSKIP(test<double>(GetParam(), onemkl::transpose::nontrans,
                                   onemkl::transpose::trans, 79, 83, 91, 103, 105, 106, alpha,
                                   beta));
    EXPECT_TRUEORSKIP(test<double>(GetParam(), onemkl::transpose::trans,
                                   onemkl::transpose::nontrans, 79, 83, 91, 103, 105, 106, alpha,
                                   beta));
    EXPECT_TRUEORSKIP(test<double>(GetParam(), onemkl::transpose::trans, onemkl::transpose::trans,
                                   79, 83, 91, 103, 105, 106, alpha, beta));
}

TEST_P(GemmUsmTests, ComplexSinglePrecision) {
    std::complex<float> alpha(2.0, -0.5);
    std::complex<float> beta(3.0, -1.5);
    EXPECT_TRUEORSKIP(test<std::complex<float>>(GetParam(), onemkl::transpose::nontrans,
                                                onemkl::transpose::nontrans, 79, 83, 91, 103, 105,
                                                106, alpha, beta));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(GetParam(), onemkl::transpose::nontrans,
                                                onemkl::transpose::trans, 79, 83, 91, 103, 105, 106,
                                                alpha, beta));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(GetParam(), onemkl::transpose::trans,
                                                onemkl::transpose::nontrans, 79, 83, 91, 103, 105,
                                                106, alpha, beta));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(GetParam(), onemkl::transpose::trans,
                                                onemkl::transpose::trans, 79, 83, 91, 103, 105, 106,
                                                alpha, beta));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(GetParam(), onemkl::transpose::nontrans,
                                                onemkl::transpose::conjtrans, 79, 83, 91, 103, 105,
                                                106, alpha, beta));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(GetParam(), onemkl::transpose::trans,
                                                onemkl::transpose::conjtrans, 79, 83, 91, 103, 105,
                                                106, alpha, beta));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(GetParam(), onemkl::transpose::conjtrans,
                                                onemkl::transpose::nontrans, 79, 83, 91, 103, 105,
                                                106, alpha, beta));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(GetParam(), onemkl::transpose::conjtrans,
                                                onemkl::transpose::trans, 79, 83, 91, 103, 105, 106,
                                                alpha, beta));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(GetParam(), onemkl::transpose::conjtrans,
                                                onemkl::transpose::conjtrans, 79, 83, 91, 103, 105,
                                                106, alpha, beta));
}

TEST_P(GemmUsmTests, ComplexDoublePrecision) {
    std::complex<double> alpha(2.0, -0.5);
    std::complex<double> beta(3.0, -1.5);
    EXPECT_TRUEORSKIP(test<std::complex<double>>(GetParam(), onemkl::transpose::nontrans,
                                                 onemkl::transpose::nontrans, 79, 83, 91, 103, 105,
                                                 106, alpha, beta));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(GetParam(), onemkl::transpose::nontrans,
                                                 onemkl::transpose::trans, 79, 83, 91, 103, 105,
                                                 106, alpha, beta));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(GetParam(), onemkl::transpose::trans,
                                                 onemkl::transpose::nontrans, 79, 83, 91, 103, 105,
                                                 106, alpha, beta));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(GetParam(), onemkl::transpose::trans,
                                                 onemkl::transpose::trans, 79, 83, 91, 103, 105,
                                                 106, alpha, beta));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(GetParam(), onemkl::transpose::nontrans,
                                                 onemkl::transpose::conjtrans, 79, 83, 91, 103, 105,
                                                 106, alpha, beta));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(GetParam(), onemkl::transpose::trans,
                                                 onemkl::transpose::conjtrans, 79, 83, 91, 103, 105,
                                                 106, alpha, beta));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(GetParam(), onemkl::transpose::conjtrans,
                                                 onemkl::transpose::nontrans, 79, 83, 91, 103, 105,
                                                 106, alpha, beta));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(GetParam(), onemkl::transpose::conjtrans,
                                                 onemkl::transpose::trans, 79, 83, 91, 103, 105,
                                                 106, alpha, beta));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(GetParam(), onemkl::transpose::conjtrans,
                                                 onemkl::transpose::conjtrans, 79, 83, 91, 103, 105,
                                                 106, alpha, beta));
}

INSTANTIATE_TEST_SUITE_P(GemmUsmTestSuite, GemmUsmTests, ::testing::ValuesIn(devices),
                         ::DeviceNamePrint());

} // anonymous namespace
