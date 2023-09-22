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

#include "skip_ahead_test.hpp"

#include <gtest/gtest.h>

extern std::vector<sycl::device*> devices;

namespace {

class Philox4x32x10DeviceSkipAheadTests : public ::testing::TestWithParam<sycl::device*> {};

// class Philox4x32x10DeviceSkipAheadExTests : public ::testing::TestWithParam<sycl::device*> {};

TEST_P(Philox4x32x10DeviceSkipAheadTests, BinaryPrecision) {
    rng_device_test<skip_ahead_test<oneapi::mkl::rng::device::philox4x32x10<1>>> test;
    EXPECT_TRUEORSKIP((test(GetParam())));
}

INSTANTIATE_TEST_SUITE_P(Philox4x32x10DeviceSkipAheadTestSuite, Philox4x32x10DeviceSkipAheadTests,
                         ::testing::ValuesIn(devices), ::DeviceNamePrint());

}
