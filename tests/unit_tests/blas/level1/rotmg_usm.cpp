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

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif
#include "cblas.h"
#include "oneapi/mkl/detail/config.hpp"
#include "oneapi/mkl.hpp"
#include "onemkl_blas_helper.hpp"
#include "reference_blas_templates.hpp"
#include "test_common.hpp"
#include "test_helper.hpp"

#include <gtest/gtest.h>

using namespace sycl;
using std::vector;

extern std::vector<sycl::device *> devices;

namespace {

template <typename fp, usm::alloc alloc_type = usm::alloc::shared>
int test(device *dev, oneapi::mkl::layout layout) {
    // Catch asynchronous exceptions.
    auto exception_handler = [](exception_list exceptions) {
        for (std::exception_ptr const &e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (exception const &e) {
                std::cout << "Caught asynchronous SYCL exception during ROTMG:\n"
                          << e.what() << std::endl;
                print_error_code(e);
            }
        }
    };

    queue main_queue(*dev, exception_handler);
    context cxt = main_queue.get_context();
    event done;
    std::vector<event> dependencies;

    // Prepare data.
    auto ua = usm_allocator<fp, usm::alloc::shared, 64>(cxt, *dev);
    vector<fp, decltype(ua)> param(5, fp(0), ua), param_ref(5, fp(0), ua);
    fp d1, d2, x1, y1, d1_ref, d2_ref, x1_ref;

    d1 = rand_scalar<fp>();
    d1 = abs(d1);
    d2 = rand_scalar<fp>();
    x1 = rand_scalar<fp>();
    y1 = rand_scalar<fp>();
    d1_ref = d1;
    d2_ref = d2;
    x1_ref = x1;

    fp *d1_p, *d2_p, *x1_p;
    if constexpr (alloc_type == usm::alloc::device) {
        d1_p = (fp *)oneapi::mkl::malloc_device(64, sizeof(fp), *dev, cxt);
        d2_p = (fp *)oneapi::mkl::malloc_device(64, sizeof(fp), *dev, cxt);
        x1_p = (fp *)oneapi::mkl::malloc_device(64, sizeof(fp), *dev, cxt);
    }
    else if constexpr (alloc_type == usm::alloc::shared) {
        d1_p = (fp *)oneapi::mkl::malloc_shared(64, sizeof(fp), *dev, cxt);
        d2_p = (fp *)oneapi::mkl::malloc_shared(64, sizeof(fp), *dev, cxt);
        x1_p = (fp *)oneapi::mkl::malloc_shared(64, sizeof(fp), *dev, cxt);
    }
    else {
        throw std::runtime_error("Bad alloc_type");
    }
    main_queue.memcpy(d1_p, &d1, sizeof(fp));
    main_queue.memcpy(d2_p, &d2, sizeof(fp));
    main_queue.memcpy(x1_p, &x1, sizeof(fp));
    main_queue.wait();

    // Call Reference ROTMG.

    ::rotmg(&d1_ref, &d2_ref, &x1_ref, &y1, (fp *)param_ref.data());

    // Call DPC++ ROTMG.

    try {
#ifdef CALL_RT_API
        switch (layout) {
            case oneapi::mkl::layout::column_major:
                done = oneapi::mkl::blas::column_major::rotmg(main_queue, d1_p, d2_p, x1_p, y1,
                                                              param.data(), dependencies);
                break;
            case oneapi::mkl::layout::row_major:
                done = oneapi::mkl::blas::row_major::rotmg(main_queue, d1_p, d2_p, x1_p, y1,
                                                           param.data(), dependencies);
                break;
            default: break;
        }
        done.wait();
#else
        switch (layout) {
            case oneapi::mkl::layout::column_major:
                TEST_RUN_CT_SELECT(main_queue, oneapi::mkl::blas::column_major::rotmg, d1_p, d2_p,
                                   x1_p, y1, param.data(), dependencies);
                break;
            case oneapi::mkl::layout::row_major:
                TEST_RUN_CT_SELECT(main_queue, oneapi::mkl::blas::row_major::rotmg, d1_p, d2_p,
                                   x1_p, y1, param.data(), dependencies);
                break;
            default: break;
        }
        main_queue.wait();
#endif
    }
    catch (exception const &e) {
        std::cout << "Caught synchronous SYCL exception during ROTMG:\n" << e.what() << std::endl;
        print_error_code(e);
    }

    catch (const oneapi::mkl::unimplemented &e) {
        return test_skipped;
    }

    catch (const std::runtime_error &error) {
        std::cout << "Error raised during execution of ROTMG:\n" << error.what() << std::endl;
    }

    // Compare the results of reference implementation and DPC++ implementation.

    bool good_d1 = check_equal_ptr(main_queue, d1_p, d1_ref, 1, std::cout);
    bool good_d2 = check_equal_ptr(main_queue, d2_p, d2_ref, 1, std::cout);
    bool good_x1 = check_equal_ptr(main_queue, x1_p, x1_ref, 1, std::cout);
    bool good_param = check_equal_vector(param, param_ref, 5, 1, 1, std::cout);
    bool good = good_d1 && good_d2 && good_x1 && good_param;

    oneapi::mkl::free_usm(d1_p, cxt);
    oneapi::mkl::free_usm(d2_p, cxt);
    oneapi::mkl::free_usm(x1_p, cxt);

    return (int)good;
}

class RotmgUsmTests
        : public ::testing::TestWithParam<std::tuple<sycl::device *, oneapi::mkl::layout>> {};

TEST_P(RotmgUsmTests, RealSinglePrecision) {
    EXPECT_TRUEORSKIP(test<float>(std::get<0>(GetParam()), std::get<1>(GetParam())));
    EXPECT_TRUEORSKIP(
        (test<float, usm::alloc::device>(std::get<0>(GetParam()), std::get<1>(GetParam()))));
}
TEST_P(RotmgUsmTests, RealDoublePrecision) {
    EXPECT_TRUEORSKIP(test<double>(std::get<0>(GetParam()), std::get<1>(GetParam())));
    EXPECT_TRUEORSKIP(
        (test<double, usm::alloc::device>(std::get<0>(GetParam()), std::get<1>(GetParam()))));
}

INSTANTIATE_TEST_SUITE_P(RotmgUsmTestSuite, RotmgUsmTests,
                         ::testing::Combine(testing::ValuesIn(devices),
                                            testing::Values(oneapi::mkl::layout::column_major,
                                                            oneapi::mkl::layout::row_major)),
                         ::LayoutDeviceNamePrint());

} // anonymous namespace
