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
#include "oneapi/mkl/detail/config.hpp"
#include "oneapi/mkl.hpp"
#include "onemkl_blas_helper.hpp"
#include "reference_blas_templates.hpp"
#include "test_common.hpp"
#include "test_helper.hpp"

#include <gtest/gtest.h>

using namespace cl::sycl;
using std::vector;

extern std::vector<cl::sycl::device> devices;

namespace {

template <typename fp, typename fp_scalar>
int test(const device& dev, oneapi::mkl::uplo upper_lower, oneapi::mkl::transpose trans, int n,
         int k, int lda, int ldb, int ldc, fp alpha, fp_scalar beta) {
    // Catch asynchronous exceptions.
    auto exception_handler = [](exception_list exceptions) {
        for (std::exception_ptr const& e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (exception const& e) {
                std::cout << "Caught asynchronous SYCL exception during HER2K:\n"
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
    auto ua = usm_allocator<fp, usm::alloc::shared, 64>(cxt, dev);
    vector<fp, decltype(ua)> A(ua), B(ua), C(ua);
    rand_matrix(A, trans, n, k, lda);
    rand_matrix(B, trans, n, k, ldb);
    rand_matrix(C, oneapi::mkl::transpose::nontrans, n, n, ldc);

    auto C_ref = C;

    // Call Reference HER2K.
    const int n_ref = n, k_ref = k;
    const int lda_ref = lda, ldb_ref = ldb, ldc_ref = ldc;

    using fp_ref = typename ref_type_info<fp>::type;
    using fp_scalar_mkl = typename ref_type_info<fp_scalar>::type;

    ::her2k(convert_to_cblas_uplo(upper_lower), convert_to_cblas_trans(trans), &n_ref, &k_ref,
            (fp_ref*)&alpha, (fp_ref*)A.data(), &lda_ref, (fp_ref*)B.data(), &ldb_ref,
            (fp_scalar_mkl*)&beta, (fp_ref*)C_ref.data(), &ldc_ref);

    // Call DPC++ HER2K.

    try {
#ifdef CALL_RT_API
        done = oneapi::mkl::blas::her2k(main_queue, upper_lower, trans, n, k, alpha, A.data(), lda,
                                        B.data(), ldb, beta, C.data(), ldc, dependencies);
        done.wait();
#else
        TEST_RUN_CT(main_queue, oneapi::mkl::blas::her2k,
                    (main_queue, upper_lower, trans, n, k, alpha, A.data(), lda, B.data(), ldb,
                     beta, C.data(), ldc, dependencies));
        main_queue.wait();
#endif
    }
    catch (exception const& e) {
        std::cout << "Caught synchronous SYCL exception during HER2K:\n"
                  << e.what() << std::endl
                  << "OpenCL status: " << e.get_cl_code() << std::endl;
    }

    catch (const oneapi::mkl::backend_unsupported_exception& e) {
        return test_skipped;
    }

    catch (const std::runtime_error& error) {
        std::cout << "Error raised during execution of HER2K:\n" << error.what() << std::endl;
    }

    // Compare the results of reference implementation and DPC++ implementation.

    bool good = check_equal_matrix(C, C_ref, n, n, ldc, 10 * std::max(n, k), std::cout);

    return (int)good;
}

class Her2kUsmTests : public ::testing::TestWithParam<cl::sycl::device> {};

TEST_P(Her2kUsmTests, ComplexSinglePrecision) {
    std::complex<float> alpha(2.0, -0.5);
    float beta(1.0);
    EXPECT_TRUEORSKIP((test<std::complex<float>, float>(GetParam(), oneapi::mkl::uplo::lower,
                                                        oneapi::mkl::transpose::nontrans, 72, 27,
                                                        101, 102, 103, alpha, beta)));
    EXPECT_TRUEORSKIP((test<std::complex<float>, float>(GetParam(), oneapi::mkl::uplo::upper,
                                                        oneapi::mkl::transpose::nontrans, 72, 27,
                                                        101, 102, 103, alpha, beta)));
    EXPECT_TRUEORSKIP((test<std::complex<float>, float>(GetParam(), oneapi::mkl::uplo::lower,
                                                        oneapi::mkl::transpose::conjtrans, 72, 27,
                                                        101, 102, 103, alpha, beta)));
    EXPECT_TRUEORSKIP((test<std::complex<float>, float>(GetParam(), oneapi::mkl::uplo::upper,
                                                        oneapi::mkl::transpose::conjtrans, 72, 27,
                                                        101, 102, 103, alpha, beta)));
}
TEST_P(Her2kUsmTests, ComplexDoublePrecision) {
    std::complex<double> alpha(2.0, -0.5);
    double beta(1.0);
    EXPECT_TRUEORSKIP((test<std::complex<double>, double>(GetParam(), oneapi::mkl::uplo::lower,
                                                          oneapi::mkl::transpose::nontrans, 72, 27,
                                                          101, 102, 103, alpha, beta)));
    EXPECT_TRUEORSKIP((test<std::complex<double>, double>(GetParam(), oneapi::mkl::uplo::upper,
                                                          oneapi::mkl::transpose::nontrans, 72, 27,
                                                          101, 102, 103, alpha, beta)));
    EXPECT_TRUEORSKIP((test<std::complex<double>, double>(GetParam(), oneapi::mkl::uplo::lower,
                                                          oneapi::mkl::transpose::conjtrans, 72, 27,
                                                          101, 102, 103, alpha, beta)));
    EXPECT_TRUEORSKIP((test<std::complex<double>, double>(GetParam(), oneapi::mkl::uplo::upper,
                                                          oneapi::mkl::transpose::conjtrans, 72, 27,
                                                          101, 102, 103, alpha, beta)));
}

INSTANTIATE_TEST_SUITE_P(Her2kUsmTestSuite, Her2kUsmTests, ::testing::ValuesIn(devices),
                         ::DeviceNamePrint());

} // anonymous namespace
