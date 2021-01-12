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

#include "statistics_check_test.hpp"

#include <gtest/gtest.h>

extern std::vector<cl::sycl::device*> devices;

namespace {

class PoissonIcdfTests : public ::testing::TestWithParam<cl::sycl::device*> {};

TEST_P(PoissonIcdfTests, IntegerPrecision) {
    rng_test<
        statistics_test<oneapi::mkl::rng::poisson<
                            std::int32_t, oneapi::mkl::rng::poisson_method::gaussian_icdf_based>,
                        oneapi::mkl::rng::philox4x32x10>>
        test1;
    EXPECT_TRUEORSKIP((test1(GetParam(), N_GEN, POISSON_ARGS)));
    rng_test<
        statistics_test<oneapi::mkl::rng::poisson<
                            std::int32_t, oneapi::mkl::rng::poisson_method::gaussian_icdf_based>,
                        oneapi::mkl::rng::mrg32k3a>>
        test2;
    EXPECT_TRUEORSKIP((test2(GetParam(), N_GEN, POISSON_ARGS)));
}

TEST_P(PoissonIcdfTests, UnsignedIntegerPrecision) {
    rng_test<
        statistics_test<oneapi::mkl::rng::poisson<
                            std::uint32_t, oneapi::mkl::rng::poisson_method::gaussian_icdf_based>,
                        oneapi::mkl::rng::philox4x32x10>>
        test1;
    EXPECT_TRUEORSKIP((test1(GetParam(), N_GEN, POISSON_ARGS)));
    rng_test<
        statistics_test<oneapi::mkl::rng::poisson<
                            std::int32_t, oneapi::mkl::rng::poisson_method::gaussian_icdf_based>,
                        oneapi::mkl::rng::mrg32k3a>>
        test2;
    EXPECT_TRUEORSKIP((test2(GetParam(), N_GEN, POISSON_ARGS)));
}

INSTANTIATE_TEST_SUITE_P(PoissonIcdfTestSuite, PoissonIcdfTests, ::testing::ValuesIn(devices),
                         ::DeviceNamePrint());

} // anonymous namespace
