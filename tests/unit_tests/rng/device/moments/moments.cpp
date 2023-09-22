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

#include "moments.hpp"

#include <gtest/gtest.h>

extern std::vector<sycl::device*> devices;

namespace {

class Philox4x32x10DeviceMomentsTests : public ::testing::TestWithParam<sycl::device*> {};

TEST_P(Philox4x32x10DeviceMomentsTests, RealSinglePrecision) {
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::philox4x32x10<1>,
    oneapi::mkl::rng::device::uniform<float, oneapi::mkl::rng::device::uniform_method::standard>>> test1;
    EXPECT_TRUEORSKIP((test1(GetParam())));
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::philox4x32x10<2>,
    oneapi::mkl::rng::device::uniform<float, oneapi::mkl::rng::device::uniform_method::standard>>> test2;
    EXPECT_TRUEORSKIP((test2(GetParam())));
}

TEST_P(Philox4x32x10DeviceMomentsTests, RealDoublePrecision) {
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::philox4x32x10<1>,
    oneapi::mkl::rng::device::uniform<double, oneapi::mkl::rng::device::uniform_method::standard>>> test;
    EXPECT_TRUEORSKIP((test(GetParam())));
}

INSTANTIATE_TEST_SUITE_P(Philox4x32x10DeviceMomentsTestSuite, Philox4x32x10DeviceMomentsTests,
                         ::testing::ValuesIn(devices), ::DeviceNamePrint());

} // anonymous namespace
