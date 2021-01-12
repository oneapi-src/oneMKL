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
#include <complex>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

#include <CL/sycl.hpp>
#include "cblas.h"
#include "oneapi/mkl.hpp"
#include "oneapi/mkl/detail/config.hpp"
#include "onemkl_blas_helper.hpp"
#include "reference_blas_templates.hpp"
#include "test_common.hpp"
#include "test_helper.hpp"

#include <gtest/gtest.h>

using namespace cl::sycl;
using std::vector;

extern std::vector<cl::sycl::device *> devices;

namespace {

template <typename fp>
int test(device *dev, oneapi::mkl::layout layout, int m, int n, fp alpha, int incx, int incy,
         int lda) {
    // Catch asynchronous exceptions.
    auto exception_handler = [](exception_list exceptions) {
        for (std::exception_ptr const &e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (exception const &e) {
                std::cout << "Caught asynchronous SYCL exception during GER:\n"
                          << e.what() << std::endl
                          << "OpenCL status: " << e.get_cl_code() << std::endl;
            }
        }
    };

    queue main_queue(*dev, exception_handler);
    context cxt = main_queue.get_context();
    event done;
    std::vector<event> dependencies;

    // Prepare data.
    auto ua = usm_allocator<fp, usm::alloc::shared, 64>(cxt, *dev);
    vector<fp, decltype(ua)> x(ua), y(ua), A(ua);

    rand_vector(x, m, incx);
    rand_vector(y, n, incy);
    rand_matrix(A, layout, oneapi::mkl::transpose::nontrans, m, n, lda);

    auto A_ref = A;

    // Call Reference GER.
    const int m_ref = m, n_ref = n, incx_ref = incx, incy_ref = incy, lda_ref = lda;
    using fp_ref = typename ref_type_info<fp>::type;

    ::ger(convert_to_cblas_layout(layout), &m_ref, &n_ref, (fp_ref *)&alpha, (fp_ref *)x.data(),
          &incx_ref, (fp_ref *)y.data(), &incy_ref, (fp_ref *)A_ref.data(), &lda_ref);

    // Call DPC++ GER.

    try {
#ifdef CALL_RT_API
        switch (layout) {
            case oneapi::mkl::layout::column_major:
                done = oneapi::mkl::blas::column_major::ger(main_queue, m, n, alpha, x.data(), incx,
                                                            y.data(), incy, A.data(), lda,
                                                            dependencies);
                break;
            case oneapi::mkl::layout::row_major:
                done =
                    oneapi::mkl::blas::row_major::ger(main_queue, m, n, alpha, x.data(), incx,
                                                      y.data(), incy, A.data(), lda, dependencies);
                break;
            default: break;
        }
        done.wait();
#else
        switch (layout) {
            case oneapi::mkl::layout::column_major:
                TEST_RUN_CT_SELECT(main_queue, oneapi::mkl::blas::column_major::ger, m, n, alpha,
                                   x.data(), incx, y.data(), incy, A.data(), lda, dependencies);
                break;
            case oneapi::mkl::layout::row_major:
                TEST_RUN_CT_SELECT(main_queue, oneapi::mkl::blas::row_major::ger, m, n, alpha,
                                   x.data(), incx, y.data(), incy, A.data(), lda, dependencies);
                break;
            default: break;
        }
        main_queue.wait();
#endif
    }
    catch (exception const &e) {
        std::cout << "Caught synchronous SYCL exception during GER:\n"
                  << e.what() << std::endl
                  << "OpenCL status: " << e.get_cl_code() << std::endl;
    }

    catch (const oneapi::mkl::unimplemented &e) {
        return test_skipped;
    }

    catch (const std::runtime_error &error) {
        std::cout << "Error raised during execution of GER:\n" << error.what() << std::endl;
    }

    // Compare the results of reference implementation and DPC++ implementation.

    bool good = check_equal_matrix(A, A_ref, layout, m, n, lda, std::max<int>(m, n), std::cout);

    return (int)good;
}

class GerUsmTests
        : public ::testing::TestWithParam<std::tuple<cl::sycl::device *, oneapi::mkl::layout>> {};

TEST_P(GerUsmTests, RealSinglePrecision) {
    float alpha(2.0);
    EXPECT_TRUEORSKIP(
        test<float>(std::get<0>(GetParam()), std::get<1>(GetParam()), 25, 30, alpha, 2, 3, 42));
    EXPECT_TRUEORSKIP(
        test<float>(std::get<0>(GetParam()), std::get<1>(GetParam()), 25, 30, alpha, -2, -3, 42));
    EXPECT_TRUEORSKIP(
        test<float>(std::get<0>(GetParam()), std::get<1>(GetParam()), 25, 30, alpha, 1, 1, 42));
}
TEST_P(GerUsmTests, RealDoublePrecision) {
    double alpha(2.0);
    EXPECT_TRUEORSKIP(
        test<double>(std::get<0>(GetParam()), std::get<1>(GetParam()), 25, 30, alpha, 2, 3, 42));
    EXPECT_TRUEORSKIP(
        test<double>(std::get<0>(GetParam()), std::get<1>(GetParam()), 25, 30, alpha, -2, -3, 42));
    EXPECT_TRUEORSKIP(
        test<double>(std::get<0>(GetParam()), std::get<1>(GetParam()), 25, 30, alpha, 1, 1, 42));
}

INSTANTIATE_TEST_SUITE_P(GerUsmTestSuite, GerUsmTests,
                         ::testing::Combine(testing::ValuesIn(devices),
                                            testing::Values(oneapi::mkl::layout::column_major,
                                                            oneapi::mkl::layout::row_major)),
                         ::LayoutDeviceNamePrint());

} // anonymous namespace
