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

extern std::vector<sycl::device*> devices;

namespace {

template <typename fp>
int test(device* dev, oneapi::mkl::layout layout) {
    // Prepare data.
    fp d1, d2, x1, y1, d1_ref, d2_ref, x1_ref;
    vector<fp> param(5, fp(0)), param_ref(5, fp(0));

    d1 = rand_scalar<fp>();
    d1 = abs(d1);
    d2 = rand_scalar<fp>();
    x1 = rand_scalar<fp>();
    y1 = rand_scalar<fp>();
    d1_ref = d1;
    d2_ref = d2;
    x1_ref = x1;

    // Call Reference ROTMG.

    ::rotmg(&d1_ref, &d2_ref, &x1_ref, &y1, (fp*)param_ref.data());

    // Call DPC++ ROTMG.

    // Catch asynchronous exceptions.
    auto exception_handler = [](exception_list exceptions) {
        for (std::exception_ptr const& e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (exception const& e) {
                std::cout << "Caught asynchronous SYCL exception during ROTMG:\n"
                          << e.what() << std::endl;
                print_error_code(e);
            }
        }
    };

    queue main_queue(*dev, exception_handler);

    buffer<fp, 1> d1_buffer(&d1, range<1>(1));
    buffer<fp, 1> d2_buffer(&d2, range<1>(1));
    buffer<fp, 1> x1_buffer(&x1, range<1>(1));
    buffer<fp, 1> param_buffer = make_buffer(param);
    try {
#ifdef CALL_RT_API
        switch (layout) {
            case oneapi::mkl::layout::col_major:
                oneapi::mkl::blas::column_major::rotmg(main_queue, d1_buffer, d2_buffer, x1_buffer,
                                                       y1, param_buffer);
                break;
            case oneapi::mkl::layout::row_major:
                oneapi::mkl::blas::row_major::rotmg(main_queue, d1_buffer, d2_buffer, x1_buffer, y1,
                                                    param_buffer);
                break;
            default: break;
        }
#else
        switch (layout) {
            case oneapi::mkl::layout::col_major:
                TEST_RUN_BLAS_CT_SELECT(main_queue, oneapi::mkl::blas::column_major::rotmg,
                                        d1_buffer, d2_buffer, x1_buffer, y1, param_buffer);
                break;
            case oneapi::mkl::layout::row_major:
                TEST_RUN_BLAS_CT_SELECT(main_queue, oneapi::mkl::blas::row_major::rotmg, d1_buffer,
                                        d2_buffer, x1_buffer, y1, param_buffer);
                break;
            default: break;
        }
#endif
    }
    catch (exception const& e) {
        std::cout << "Caught synchronous SYCL exception during ROTMG:\n" << e.what() << std::endl;
        print_error_code(e);
    }

    catch (const oneapi::mkl::unimplemented& e) {
        return test_skipped;
    }

    catch (const std::runtime_error& error) {
        std::cout << "Error raised during execution of ROTMG:\n" << error.what() << std::endl;
    }

    // Compare the results of reference implementation and DPC++ implementation.

    int error_mag = 50;

    auto d1_accessor = d1_buffer.get_host_access(read_only);
    bool good_d1 = check_equal(d1_accessor[0], d1_ref, error_mag, std::cout);
    auto d2_accessor = d2_buffer.get_host_access(read_only);
    bool good_d2 = check_equal(d2_accessor[0], d2_ref, error_mag, std::cout);
    auto x1_accessor = x1_buffer.get_host_access(read_only);
    bool good_x1 = check_equal(x1_accessor[0], x1_ref, error_mag, std::cout);
    auto param_accessor = param_buffer.get_host_access(read_only);

    constexpr fp unit_matrix = -2;
    constexpr fp rescaled_matrix = -1;
    constexpr fp sltc_matrix = 0;
    constexpr fp clts_matrix = 1;

    fp flag = param_accessor[0];
    fp h11 = param_accessor[1];
    fp h12 = param_accessor[3];
    fp h21 = param_accessor[2];
    fp h22 = param_accessor[4];

    fp flag_ref = param_ref[0];
    fp h11_ref = param_ref[1];
    fp h12_ref = param_ref[3];
    fp h21_ref = param_ref[2];
    fp h22_ref = param_ref[4];

    bool flag_good = (flag_ref == flag);
    bool h11_good = true;
    bool h12_good = true;
    bool h21_good = true;
    bool h22_good = true;

    /* Some values of param have to be ignored depending on the flag value since they are
     * implementation defined */
    if (flag_ref != unit_matrix) {
        if (flag_ref == sltc_matrix) {
            h12_good = check_equal(h12, h12_ref, error_mag, std::cout);
            h21_good = check_equal(h21, h21_ref, error_mag, std::cout);
        }
        else if (flag_ref == clts_matrix) {
            h11_good = check_equal(h11, h11_ref, error_mag, std::cout);
            h22_good = check_equal(h22, h22_ref, error_mag, std::cout);
        }
        else {
            flag_good = flag_good && (flag == rescaled_matrix);
            h11_good = check_equal(h11, h11_ref, error_mag, std::cout);
            h12_good = check_equal(h12, h12_ref, error_mag, std::cout);
            h21_good = check_equal(h21, h21_ref, error_mag, std::cout);
            h22_good = check_equal(h22, h22_ref, error_mag, std::cout);
        }
    }

    bool good =
        good_d1 && good_d2 && good_x1 && flag_good && h11_good && h12_good && h21_good && h22_good;

    return (int)good;
}

class RotmgTests : public ::testing::TestWithParam<std::tuple<sycl::device*, oneapi::mkl::layout>> {
};

TEST_P(RotmgTests, RealSinglePrecision) {
    EXPECT_TRUEORSKIP(test<float>(std::get<0>(GetParam()), std::get<1>(GetParam())));
}
TEST_P(RotmgTests, RealDoublePrecision) {
    CHECK_DOUBLE_ON_DEVICE(std::get<0>(GetParam()));

    EXPECT_TRUEORSKIP(test<double>(std::get<0>(GetParam()), std::get<1>(GetParam())));
}

INSTANTIATE_TEST_SUITE_P(RotmgTestSuite, RotmgTests,
                         ::testing::Combine(testing::ValuesIn(devices),
                                            testing::Values(oneapi::mkl::layout::col_major,
                                                            oneapi::mkl::layout::row_major)),
                         ::LayoutDeviceNamePrint());

} // anonymous namespace
