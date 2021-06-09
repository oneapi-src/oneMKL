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

#include <cstdint>
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

extern std::vector<cl::sycl::device *> devices;

namespace {

template <typename fp, typename fp_scalar>
int test(device *dev, oneapi::mkl::layout layout) {
    // Catch asynchronous exceptions.
    auto exception_handler = [](exception_list exceptions) {
        for (std::exception_ptr const &e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (exception const &e) {
                std::cout << "Caught asynchronous SYCL exception during ROTG:\n"
                          << e.what() << std::endl
                          << "OpenCL status: " << e.what() << std::endl;
            }
        }
    };

    queue main_queue(*dev, exception_handler);
    context cxt = main_queue.get_context();
    event done;
    std::vector<event> dependencies;

    // Prepare data.

    fp a, b, s, a_ref, b_ref, s_ref;
    fp_scalar c, c_ref;

    a = rand_scalar<fp>();
    b = rand_scalar<fp>();
    s = rand_scalar<fp>();
    c = rand_scalar<fp_scalar>();
    a_ref = a;
    b_ref = b;
    s_ref = s;
    c_ref = c;

    // Call Reference ROTG.
    using fp_ref = typename ref_type_info<fp>::type;

    ::rotg((fp_ref *)&a_ref, (fp_ref *)&b_ref, (fp_scalar *)&c_ref, (fp_ref *)&s_ref);

    // Call DPC++ ROTG.
    fp *a_p = (fp *)oneapi::mkl::malloc_shared(64, sizeof(fp), *dev, cxt);
    fp *b_p = (fp *)oneapi::mkl::malloc_shared(64, sizeof(fp), *dev, cxt);
    fp *s_p = (fp *)oneapi::mkl::malloc_shared(64, sizeof(fp), *dev, cxt);
    fp_scalar *c_p = (fp_scalar *)oneapi::mkl::malloc_shared(64, sizeof(fp_scalar), *dev, cxt);

    a_p[0] = a;
    b_p[0] = b;
    s_p[0] = s;
    c_p[0] = c;

    try {
#ifdef CALL_RT_API
        switch (layout) {
            case oneapi::mkl::layout::column_major:
                done = oneapi::mkl::blas::column_major::rotg(main_queue, a_p, b_p, c_p, s_p,
                                                             dependencies);
                break;
            case oneapi::mkl::layout::row_major:
                done = oneapi::mkl::blas::row_major::rotg(main_queue, a_p, b_p, c_p, s_p,
                                                          dependencies);
                break;
            default: break;
        }
        done.wait();
#else
        switch (layout) {
            case oneapi::mkl::layout::column_major:
                TEST_RUN_CT_SELECT(main_queue, oneapi::mkl::blas::column_major::rotg, a_p, b_p, c_p,
                                   s_p, dependencies);
                break;
            case oneapi::mkl::layout::row_major:
                TEST_RUN_CT_SELECT(main_queue, oneapi::mkl::blas::row_major::rotg, a_p, b_p, c_p,
                                   s_p, dependencies);
                break;
            default: break;
        }
        main_queue.wait();
#endif
    }
    catch (exception const &e) {
        std::cout << "Caught synchronous SYCL exception during ROTG:\n"
                  << e.what() << std::endl
                  << "OpenCL status: " << e.what() << std::endl;
    }

    catch (const oneapi::mkl::unimplemented &e) {
        return test_skipped;
    }

    catch (const std::runtime_error &error) {
        std::cout << "Error raised during execution of ROTG:\n" << error.what() << std::endl;
    }

    // Compare the results of reference implementation and DPC++ implementation.

    bool good_a = check_equal(a_p[0], a_ref, 4, std::cout);
    bool good_b = check_equal(b_p[0], b_ref, 4, std::cout);
    bool good_s = check_equal(s_p[0], s_ref, 4, std::cout);
    bool good_c = check_equal(c_p[0], c_ref, 4, std::cout);

    bool good = good_a && good_b && good_c && good_s;

    oneapi::mkl::free_shared(a_p, cxt);
    oneapi::mkl::free_shared(b_p, cxt);
    oneapi::mkl::free_shared(s_p, cxt);
    oneapi::mkl::free_shared(c_p, cxt);

    return (int)good;
}

class RotgUsmTests
        : public ::testing::TestWithParam<std::tuple<cl::sycl::device *, oneapi::mkl::layout>> {};

TEST_P(RotgUsmTests, RealSinglePrecision) {
    EXPECT_TRUEORSKIP((test<float, float>(std::get<0>(GetParam()), std::get<1>(GetParam()))));
}
TEST_P(RotgUsmTests, RealDoublePrecision) {
    EXPECT_TRUEORSKIP((test<double, double>(std::get<0>(GetParam()), std::get<1>(GetParam()))));
}
TEST_P(RotgUsmTests, ComplexSinglePrecision) {
    EXPECT_TRUEORSKIP(
        (test<std::complex<float>, float>(std::get<0>(GetParam()), std::get<1>(GetParam()))));
}
TEST_P(RotgUsmTests, ComplexDoublePrecision) {
    EXPECT_TRUEORSKIP(
        (test<std::complex<double>, double>(std::get<0>(GetParam()), std::get<1>(GetParam()))));
}

INSTANTIATE_TEST_SUITE_P(RotgUsmTestSuite, RotgUsmTests,
                         ::testing::Combine(testing::ValuesIn(devices),
                                            testing::Values(oneapi::mkl::layout::column_major,
                                                            oneapi::mkl::layout::row_major)),
                         ::LayoutDeviceNamePrint());

} // anonymous namespace
