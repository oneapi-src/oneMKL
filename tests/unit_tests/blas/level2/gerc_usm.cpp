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
#include <complex>
#include <cstdlib>
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

template <typename fp>
int test(const device &dev, int m, int n, fp alpha, int incx, int incy, int lda) {
    // Catch asynchronous exceptions.
    auto exception_handler = [](exception_list exceptions) {
        for (std::exception_ptr const &e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (exception const &e) {
                std::cout << "Caught asynchronous SYCL exception during GERC:\n"
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
    vector<fp, decltype(ua)> x(ua), y(ua), A(ua);

    rand_vector(x, m, incx);
    rand_vector(y, n, incy);
    rand_matrix(A, oneapi::mkl::transpose::nontrans, m, n, lda);

    auto A_ref = A;

    // Call Reference GERC.
    const int m_ref = m, n_ref = n, incx_ref = incx, incy_ref = incy, lda_ref = lda;
    using fp_ref = typename ref_type_info<fp>::type;

    ::gerc(&m_ref, &n_ref, (fp_ref *)&alpha, (fp_ref *)x.data(), &incx_ref, (fp_ref *)y.data(),
           &incy_ref, (fp_ref *)A_ref.data(), &lda_ref);

    // Call DPC++ GERC.

    try {
#ifdef CALL_RT_API
        done = oneapi::mkl::blas::gerc(main_queue, m, n, alpha, x.data(), incx, y.data(), incy, A.data(),
                                  lda, dependencies);
        done.wait();
#else
        TEST_RUN_CT(
            main_queue, oneapi::mkl::blas::gerc,
            (main_queue, m, n, alpha, x.data(), incx, y.data(), incy, A.data(), lda, dependencies));
        main_queue.wait();
#endif
    }
    catch (exception const &e) {
        std::cout << "Caught synchronous SYCL exception during GERC:\n"
                  << e.what() << std::endl
                  << "OpenCL status: " << e.get_cl_code() << std::endl;
    }

    catch (const oneapi::mkl::backend_unsupported_exception &e) {
        return test_skipped;
    }

    catch (const std::runtime_error &error) {
        std::cout << "Error raised during execution of GERC:\n" << error.what() << std::endl;
    }

    // Compare the results of reference implementation and DPC++ implementation.

    bool good = check_equal_matrix(A, A_ref, m, n, lda, std::max<int>(m, n), std::cout);

    return (int)good;
}

class GercUsmTests : public ::testing::TestWithParam<cl::sycl::device> {};

TEST_P(GercUsmTests, ComplexSinglePrecision) {
    std::complex<float> alpha(2.0, -0.5);
    EXPECT_TRUEORSKIP(test<std::complex<float>>(GetParam(), 25, 30, alpha, 2, 3, 42));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(GetParam(), 25, 30, alpha, -2, -3, 42));
    EXPECT_TRUEORSKIP(test<std::complex<float>>(GetParam(), 25, 30, alpha, 1, 1, 42));
}
TEST_P(GercUsmTests, ComplexDoublePrecision) {
    std::complex<double> alpha(2.0, -0.5);
    EXPECT_TRUEORSKIP(test<std::complex<double>>(GetParam(), 25, 30, alpha, 2, 3, 42));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(GetParam(), 25, 30, alpha, -2, -3, 42));
    EXPECT_TRUEORSKIP(test<std::complex<double>>(GetParam(), 25, 30, alpha, 1, 1, 42));
}

INSTANTIATE_TEST_SUITE_P(GercUsmTestSuite, GercUsmTests, ::testing::ValuesIn(devices),
                         ::DeviceNamePrint());

} // anonymous namespace
