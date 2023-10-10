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

class Philox4x32x10UniformStdDeviceMomentsTests : public ::testing::TestWithParam<sycl::device*> {};

class Philox4x32x10UniformAccDeviceMomentsTests : public ::testing::TestWithParam<sycl::device*> {};

TEST_P(Philox4x32x10UniformStdDeviceMomentsTests, RealSinglePrecision) {
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::philox4x32x10<1>,
    oneapi::mkl::rng::device::uniform<float, oneapi::mkl::rng::device::uniform_method::standard>>> test1;
    EXPECT_TRUEORSKIP((test1(GetParam())));
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::philox4x32x10<4>,
    oneapi::mkl::rng::device::uniform<float, oneapi::mkl::rng::device::uniform_method::standard>>> test2;
    EXPECT_TRUEORSKIP((test2(GetParam())));
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::philox4x32x10<16>,
    oneapi::mkl::rng::device::uniform<float, oneapi::mkl::rng::device::uniform_method::standard>>> test3;
    EXPECT_TRUEORSKIP((test3(GetParam())));
}

TEST_P(Philox4x32x10UniformStdDeviceMomentsTests, RealDoublePrecision) {
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::philox4x32x10<1>,
    oneapi::mkl::rng::device::uniform<double, oneapi::mkl::rng::device::uniform_method::standard>>> test1;
    EXPECT_TRUEORSKIP((test1(GetParam())));
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::philox4x32x10<4>,
    oneapi::mkl::rng::device::uniform<double, oneapi::mkl::rng::device::uniform_method::standard>>> test2;
    EXPECT_TRUEORSKIP((test2(GetParam())));
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::philox4x32x10<16>,
    oneapi::mkl::rng::device::uniform<double, oneapi::mkl::rng::device::uniform_method::standard>>> test3;
    EXPECT_TRUEORSKIP((test3(GetParam())));
}

TEST_P(Philox4x32x10UniformStdDeviceMomentsTests, IntegerPrecision) {
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::philox4x32x10<1>,
    oneapi::mkl::rng::device::uniform<std::int32_t, oneapi::mkl::rng::device::uniform_method::standard>>> test1;
    EXPECT_TRUEORSKIP((test1(GetParam())));
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::philox4x32x10<4>,
    oneapi::mkl::rng::device::uniform<std::int32_t, oneapi::mkl::rng::device::uniform_method::standard>>> test2;
    EXPECT_TRUEORSKIP((test2(GetParam())));
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::philox4x32x10<16>,
    oneapi::mkl::rng::device::uniform<std::int32_t, oneapi::mkl::rng::device::uniform_method::standard>>> test3;
    EXPECT_TRUEORSKIP((test3(GetParam())));
}

TEST_P(Philox4x32x10UniformStdDeviceMomentsTests, UnsignedIntegerPrecision) {
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::philox4x32x10<1>,
    oneapi::mkl::rng::device::uniform<std::uint32_t, oneapi::mkl::rng::device::uniform_method::standard>>> test1;
    EXPECT_TRUEORSKIP((test1(GetParam())));
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::philox4x32x10<4>,
    oneapi::mkl::rng::device::uniform<std::uint32_t, oneapi::mkl::rng::device::uniform_method::standard>>> test2;
    EXPECT_TRUEORSKIP((test2(GetParam())));
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::philox4x32x10<16>,
    oneapi::mkl::rng::device::uniform<std::uint32_t, oneapi::mkl::rng::device::uniform_method::standard>>> test3;
    EXPECT_TRUEORSKIP((test3(GetParam())));
}

TEST_P(Philox4x32x10UniformAccDeviceMomentsTests, RealSinglePrecision) {
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::philox4x32x10<1>,
    oneapi::mkl::rng::device::uniform<float, oneapi::mkl::rng::device::uniform_method::accurate>>> test1;
    EXPECT_TRUEORSKIP((test1(GetParam())));
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::philox4x32x10<4>,
    oneapi::mkl::rng::device::uniform<float, oneapi::mkl::rng::device::uniform_method::accurate>>> test2;
    EXPECT_TRUEORSKIP((test2(GetParam())));
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::philox4x32x10<16>,
    oneapi::mkl::rng::device::uniform<float, oneapi::mkl::rng::device::uniform_method::accurate>>> test3;
    EXPECT_TRUEORSKIP((test3(GetParam())));
}

TEST_P(Philox4x32x10UniformAccDeviceMomentsTests, RealDoublePrecision) {
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::philox4x32x10<1>,
    oneapi::mkl::rng::device::uniform<double, oneapi::mkl::rng::device::uniform_method::accurate>>> test1;
    EXPECT_TRUEORSKIP((test1(GetParam())));
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::philox4x32x10<4>,
    oneapi::mkl::rng::device::uniform<double, oneapi::mkl::rng::device::uniform_method::accurate>>> test2;
    EXPECT_TRUEORSKIP((test2(GetParam())));
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::philox4x32x10<16>,
    oneapi::mkl::rng::device::uniform<double, oneapi::mkl::rng::device::uniform_method::accurate>>> test3;
    EXPECT_TRUEORSKIP((test3(GetParam())));
}

TEST_P(Philox4x32x10UniformAccDeviceMomentsTests, IntegerPrecision) {
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::philox4x32x10<1>,
    oneapi::mkl::rng::device::uniform<std::int32_t, oneapi::mkl::rng::device::uniform_method::accurate>>> test1;
    EXPECT_TRUEORSKIP((test1(GetParam())));
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::philox4x32x10<4>,
    oneapi::mkl::rng::device::uniform<std::int32_t, oneapi::mkl::rng::device::uniform_method::accurate>>> test2;
    EXPECT_TRUEORSKIP((test2(GetParam())));
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::philox4x32x10<16>,
    oneapi::mkl::rng::device::uniform<std::int32_t, oneapi::mkl::rng::device::uniform_method::accurate>>> test3;
    EXPECT_TRUEORSKIP((test3(GetParam())));
}

TEST_P(Philox4x32x10UniformAccDeviceMomentsTests, UnsignedIntegerPrecision) {
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::philox4x32x10<1>,
    oneapi::mkl::rng::device::uniform<std::uint32_t, oneapi::mkl::rng::device::uniform_method::accurate>>> test1;
    EXPECT_TRUEORSKIP((test1(GetParam())));
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::philox4x32x10<4>,
    oneapi::mkl::rng::device::uniform<std::uint32_t, oneapi::mkl::rng::device::uniform_method::accurate>>> test2;
    EXPECT_TRUEORSKIP((test2(GetParam())));
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::philox4x32x10<16>,
    oneapi::mkl::rng::device::uniform<std::uint32_t, oneapi::mkl::rng::device::uniform_method::accurate>>> test3;
    EXPECT_TRUEORSKIP((test3(GetParam())));
}

INSTANTIATE_TEST_SUITE_P(Philox4x32x10UniformStdDeviceMomentsTestsSuite, Philox4x32x10UniformStdDeviceMomentsTests,
                         ::testing::ValuesIn(devices), ::DeviceNamePrint());

INSTANTIATE_TEST_SUITE_P(Philox4x32x10UniformAccDeviceMomentsTestsSuite, Philox4x32x10UniformAccDeviceMomentsTests,
                         ::testing::ValuesIn(devices), ::DeviceNamePrint());

class Mrg32k3aUniformStdDeviceMomentsTests : public ::testing::TestWithParam<sycl::device*> {};

class Mrg32k3aUniformAccDeviceMomentsTests : public ::testing::TestWithParam<sycl::device*> {};

TEST_P(Mrg32k3aUniformStdDeviceMomentsTests, RealSinglePrecision) {
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::mrg32k3a<1>,
    oneapi::mkl::rng::device::uniform<float, oneapi::mkl::rng::device::uniform_method::standard>>> test1;
    EXPECT_TRUEORSKIP((test1(GetParam())));
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::mrg32k3a<4>,
    oneapi::mkl::rng::device::uniform<float, oneapi::mkl::rng::device::uniform_method::standard>>> test2;
    EXPECT_TRUEORSKIP((test2(GetParam())));
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::mrg32k3a<16>,
    oneapi::mkl::rng::device::uniform<float, oneapi::mkl::rng::device::uniform_method::standard>>> test3;
    EXPECT_TRUEORSKIP((test3(GetParam())));
}

TEST_P(Mrg32k3aUniformStdDeviceMomentsTests, RealDoublePrecision) {
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::mrg32k3a<1>,
    oneapi::mkl::rng::device::uniform<double, oneapi::mkl::rng::device::uniform_method::standard>>> test1;
    EXPECT_TRUEORSKIP((test1(GetParam())));
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::mrg32k3a<4>,
    oneapi::mkl::rng::device::uniform<double, oneapi::mkl::rng::device::uniform_method::standard>>> test2;
    EXPECT_TRUEORSKIP((test2(GetParam())));
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::mrg32k3a<16>,
    oneapi::mkl::rng::device::uniform<double, oneapi::mkl::rng::device::uniform_method::standard>>> test3;
    EXPECT_TRUEORSKIP((test3(GetParam())));
}

TEST_P(Mrg32k3aUniformStdDeviceMomentsTests, IntegerPrecision) {
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::mrg32k3a<1>,
    oneapi::mkl::rng::device::uniform<std::int32_t, oneapi::mkl::rng::device::uniform_method::standard>>> test1;
    EXPECT_TRUEORSKIP((test1(GetParam())));
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::mrg32k3a<4>,
    oneapi::mkl::rng::device::uniform<std::int32_t, oneapi::mkl::rng::device::uniform_method::standard>>> test2;
    EXPECT_TRUEORSKIP((test2(GetParam())));
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::mrg32k3a<16>,
    oneapi::mkl::rng::device::uniform<std::int32_t, oneapi::mkl::rng::device::uniform_method::standard>>> test3;
    EXPECT_TRUEORSKIP((test3(GetParam())));
}

TEST_P(Mrg32k3aUniformStdDeviceMomentsTests, UnsignedIntegerPrecision) {
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::mrg32k3a<1>,
    oneapi::mkl::rng::device::uniform<std::uint32_t, oneapi::mkl::rng::device::uniform_method::standard>>> test1;
    EXPECT_TRUEORSKIP((test1(GetParam())));
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::mrg32k3a<4>,
    oneapi::mkl::rng::device::uniform<std::uint32_t, oneapi::mkl::rng::device::uniform_method::standard>>> test2;
    EXPECT_TRUEORSKIP((test2(GetParam())));
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::mrg32k3a<16>,
    oneapi::mkl::rng::device::uniform<std::uint32_t, oneapi::mkl::rng::device::uniform_method::standard>>> test3;
    EXPECT_TRUEORSKIP((test3(GetParam())));
}

TEST_P(Mrg32k3aUniformAccDeviceMomentsTests, RealSinglePrecision) {
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::mrg32k3a<1>,
    oneapi::mkl::rng::device::uniform<float, oneapi::mkl::rng::device::uniform_method::accurate>>> test1;
    EXPECT_TRUEORSKIP((test1(GetParam())));
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::mrg32k3a<4>,
    oneapi::mkl::rng::device::uniform<float, oneapi::mkl::rng::device::uniform_method::accurate>>> test2;
    EXPECT_TRUEORSKIP((test2(GetParam())));
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::mrg32k3a<16>,
    oneapi::mkl::rng::device::uniform<float, oneapi::mkl::rng::device::uniform_method::accurate>>> test3;
    EXPECT_TRUEORSKIP((test3(GetParam())));
}

TEST_P(Mrg32k3aUniformAccDeviceMomentsTests, RealDoublePrecision) {
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::mrg32k3a<1>,
    oneapi::mkl::rng::device::uniform<double, oneapi::mkl::rng::device::uniform_method::accurate>>> test1;
    EXPECT_TRUEORSKIP((test1(GetParam())));
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::mrg32k3a<4>,
    oneapi::mkl::rng::device::uniform<double, oneapi::mkl::rng::device::uniform_method::accurate>>> test2;
    EXPECT_TRUEORSKIP((test2(GetParam())));
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::mrg32k3a<16>,
    oneapi::mkl::rng::device::uniform<double, oneapi::mkl::rng::device::uniform_method::accurate>>> test3;
    EXPECT_TRUEORSKIP((test3(GetParam())));
}

TEST_P(Mrg32k3aUniformAccDeviceMomentsTests, IntegerPrecision) {
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::mrg32k3a<1>,
    oneapi::mkl::rng::device::uniform<std::int32_t, oneapi::mkl::rng::device::uniform_method::accurate>>> test1;
    EXPECT_TRUEORSKIP((test1(GetParam())));
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::mrg32k3a<4>,
    oneapi::mkl::rng::device::uniform<std::int32_t, oneapi::mkl::rng::device::uniform_method::accurate>>> test2;
    EXPECT_TRUEORSKIP((test2(GetParam())));
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::mrg32k3a<16>,
    oneapi::mkl::rng::device::uniform<std::int32_t, oneapi::mkl::rng::device::uniform_method::accurate>>> test3;
    EXPECT_TRUEORSKIP((test3(GetParam())));
}

TEST_P(Mrg32k3aUniformAccDeviceMomentsTests, UnsignedIntegerPrecision) {
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::mrg32k3a<1>,
    oneapi::mkl::rng::device::uniform<std::uint32_t, oneapi::mkl::rng::device::uniform_method::accurate>>> test1;
    EXPECT_TRUEORSKIP((test1(GetParam())));
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::mrg32k3a<4>,
    oneapi::mkl::rng::device::uniform<std::uint32_t, oneapi::mkl::rng::device::uniform_method::accurate>>> test2;
    EXPECT_TRUEORSKIP((test2(GetParam())));
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::mrg32k3a<16>,
    oneapi::mkl::rng::device::uniform<std::uint32_t, oneapi::mkl::rng::device::uniform_method::accurate>>> test3;
    EXPECT_TRUEORSKIP((test3(GetParam())));
}

INSTANTIATE_TEST_SUITE_P(Mrg32k3aUniformStdDeviceMomentsTestsSuite, Mrg32k3aUniformStdDeviceMomentsTests,
                         ::testing::ValuesIn(devices), ::DeviceNamePrint());

INSTANTIATE_TEST_SUITE_P(Mrg32k3aUniformAccDeviceMomentsTestsSuite, Mrg32k3aUniformAccDeviceMomentsTests,
                         ::testing::ValuesIn(devices), ::DeviceNamePrint());

class Mcg31m1UniformStdDeviceMomentsTests : public ::testing::TestWithParam<sycl::device*> {};

class Mcg31m1UniformAccDeviceMomentsTests : public ::testing::TestWithParam<sycl::device*> {};

TEST_P(Mcg31m1UniformStdDeviceMomentsTests, RealSinglePrecision) {
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::mcg31m1<1>,
    oneapi::mkl::rng::device::uniform<float, oneapi::mkl::rng::device::uniform_method::standard>>> test1;
    EXPECT_TRUEORSKIP((test1(GetParam())));
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::mcg31m1<4>,
    oneapi::mkl::rng::device::uniform<float, oneapi::mkl::rng::device::uniform_method::standard>>> test2;
    EXPECT_TRUEORSKIP((test2(GetParam())));
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::mcg31m1<16>,
    oneapi::mkl::rng::device::uniform<float, oneapi::mkl::rng::device::uniform_method::standard>>> test3;
    EXPECT_TRUEORSKIP((test3(GetParam())));
}

TEST_P(Mcg31m1UniformStdDeviceMomentsTests, RealDoublePrecision) {
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::mcg31m1<1>,
    oneapi::mkl::rng::device::uniform<double, oneapi::mkl::rng::device::uniform_method::standard>>> test1;
    EXPECT_TRUEORSKIP((test1(GetParam())));
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::mcg31m1<4>,
    oneapi::mkl::rng::device::uniform<double, oneapi::mkl::rng::device::uniform_method::standard>>> test2;
    EXPECT_TRUEORSKIP((test2(GetParam())));
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::mcg31m1<16>,
    oneapi::mkl::rng::device::uniform<double, oneapi::mkl::rng::device::uniform_method::standard>>> test3;
    EXPECT_TRUEORSKIP((test3(GetParam())));
}

TEST_P(Mcg31m1UniformStdDeviceMomentsTests, IntegerPrecision) {
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::mcg31m1<1>,
    oneapi::mkl::rng::device::uniform<std::int32_t, oneapi::mkl::rng::device::uniform_method::standard>>> test1;
    EXPECT_TRUEORSKIP((test1(GetParam())));
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::mcg31m1<4>,
    oneapi::mkl::rng::device::uniform<std::int32_t, oneapi::mkl::rng::device::uniform_method::standard>>> test2;
    EXPECT_TRUEORSKIP((test2(GetParam())));
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::mcg31m1<16>,
    oneapi::mkl::rng::device::uniform<std::int32_t, oneapi::mkl::rng::device::uniform_method::standard>>> test3;
    EXPECT_TRUEORSKIP((test3(GetParam())));
}

TEST_P(Mcg31m1UniformStdDeviceMomentsTests, UnsignedIntegerPrecision) {
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::mcg31m1<1>,
    oneapi::mkl::rng::device::uniform<std::uint32_t, oneapi::mkl::rng::device::uniform_method::standard>>> test1;
    EXPECT_TRUEORSKIP((test1(GetParam())));
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::mcg31m1<4>,
    oneapi::mkl::rng::device::uniform<std::uint32_t, oneapi::mkl::rng::device::uniform_method::standard>>> test2;
    EXPECT_TRUEORSKIP((test2(GetParam())));
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::mcg31m1<16>,
    oneapi::mkl::rng::device::uniform<std::uint32_t, oneapi::mkl::rng::device::uniform_method::standard>>> test3;
    EXPECT_TRUEORSKIP((test3(GetParam())));
}

TEST_P(Mcg31m1UniformAccDeviceMomentsTests, RealSinglePrecision) {
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::mcg31m1<1>,
    oneapi::mkl::rng::device::uniform<float, oneapi::mkl::rng::device::uniform_method::accurate>>> test1;
    EXPECT_TRUEORSKIP((test1(GetParam())));
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::mcg31m1<4>,
    oneapi::mkl::rng::device::uniform<float, oneapi::mkl::rng::device::uniform_method::accurate>>> test2;
    EXPECT_TRUEORSKIP((test2(GetParam())));
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::mcg31m1<16>,
    oneapi::mkl::rng::device::uniform<float, oneapi::mkl::rng::device::uniform_method::accurate>>> test3;
    EXPECT_TRUEORSKIP((test3(GetParam())));
}

TEST_P(Mcg31m1UniformAccDeviceMomentsTests, RealDoublePrecision) {
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::mcg31m1<1>,
    oneapi::mkl::rng::device::uniform<double, oneapi::mkl::rng::device::uniform_method::accurate>>> test1;
    EXPECT_TRUEORSKIP((test1(GetParam())));
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::mcg31m1<4>,
    oneapi::mkl::rng::device::uniform<double, oneapi::mkl::rng::device::uniform_method::accurate>>> test2;
    EXPECT_TRUEORSKIP((test2(GetParam())));
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::mcg31m1<16>,
    oneapi::mkl::rng::device::uniform<double, oneapi::mkl::rng::device::uniform_method::accurate>>> test3;
    EXPECT_TRUEORSKIP((test3(GetParam())));
}

TEST_P(Mcg31m1UniformAccDeviceMomentsTests, IntegerPrecision) {
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::mcg31m1<1>,
    oneapi::mkl::rng::device::uniform<std::int32_t, oneapi::mkl::rng::device::uniform_method::accurate>>> test1;
    EXPECT_TRUEORSKIP((test1(GetParam())));
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::mcg31m1<4>,
    oneapi::mkl::rng::device::uniform<std::int32_t, oneapi::mkl::rng::device::uniform_method::accurate>>> test2;
    EXPECT_TRUEORSKIP((test2(GetParam())));
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::mcg31m1<16>,
    oneapi::mkl::rng::device::uniform<std::int32_t, oneapi::mkl::rng::device::uniform_method::accurate>>> test3;
    EXPECT_TRUEORSKIP((test3(GetParam())));
}

TEST_P(Mcg31m1UniformAccDeviceMomentsTests, UnsignedIntegerPrecision) {
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::mcg31m1<1>,
    oneapi::mkl::rng::device::uniform<std::uint32_t, oneapi::mkl::rng::device::uniform_method::accurate>>> test1;
    EXPECT_TRUEORSKIP((test1(GetParam())));
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::mcg31m1<4>,
    oneapi::mkl::rng::device::uniform<std::uint32_t, oneapi::mkl::rng::device::uniform_method::accurate>>> test2;
    EXPECT_TRUEORSKIP((test2(GetParam())));
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::mcg31m1<16>,
    oneapi::mkl::rng::device::uniform<std::uint32_t, oneapi::mkl::rng::device::uniform_method::accurate>>> test3;
    EXPECT_TRUEORSKIP((test3(GetParam())));
}

INSTANTIATE_TEST_SUITE_P(Mcg31m1UniformStdDeviceMomentsTestsSuite, Mcg31m1UniformStdDeviceMomentsTests,
                         ::testing::ValuesIn(devices), ::DeviceNamePrint());

INSTANTIATE_TEST_SUITE_P(Mcg31m1UniformAccDeviceMomentsTestsSuite, Mcg31m1UniformAccDeviceMomentsTests,
                         ::testing::ValuesIn(devices), ::DeviceNamePrint());

class Mcg59UniformStdDeviceMomentsTests : public ::testing::TestWithParam<sycl::device*> {};

class Mcg59UniformAccDeviceMomentsTests : public ::testing::TestWithParam<sycl::device*> {};

TEST_P(Mcg59UniformStdDeviceMomentsTests, RealSinglePrecision) {
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::mcg59<1>,
    oneapi::mkl::rng::device::uniform<float, oneapi::mkl::rng::device::uniform_method::standard>>> test1;
    EXPECT_TRUEORSKIP((test1(GetParam())));
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::mcg59<4>,
    oneapi::mkl::rng::device::uniform<float, oneapi::mkl::rng::device::uniform_method::standard>>> test2;
    EXPECT_TRUEORSKIP((test2(GetParam())));
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::mcg59<16>,
    oneapi::mkl::rng::device::uniform<float, oneapi::mkl::rng::device::uniform_method::standard>>> test3;
    EXPECT_TRUEORSKIP((test3(GetParam())));
}

TEST_P(Mcg59UniformStdDeviceMomentsTests, RealDoublePrecision) {
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::mcg59<1>,
    oneapi::mkl::rng::device::uniform<double, oneapi::mkl::rng::device::uniform_method::standard>>> test1;
    EXPECT_TRUEORSKIP((test1(GetParam())));
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::mcg59<4>,
    oneapi::mkl::rng::device::uniform<double, oneapi::mkl::rng::device::uniform_method::standard>>> test2;
    EXPECT_TRUEORSKIP((test2(GetParam())));
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::mcg59<16>,
    oneapi::mkl::rng::device::uniform<double, oneapi::mkl::rng::device::uniform_method::standard>>> test3;
    EXPECT_TRUEORSKIP((test3(GetParam())));
}

TEST_P(Mcg59UniformStdDeviceMomentsTests, IntegerPrecision) {
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::mcg59<1>,
    oneapi::mkl::rng::device::uniform<std::int32_t, oneapi::mkl::rng::device::uniform_method::standard>>> test1;
    EXPECT_TRUEORSKIP((test1(GetParam())));
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::mcg59<4>,
    oneapi::mkl::rng::device::uniform<std::int32_t, oneapi::mkl::rng::device::uniform_method::standard>>> test2;
    EXPECT_TRUEORSKIP((test2(GetParam())));
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::mcg59<16>,
    oneapi::mkl::rng::device::uniform<std::int32_t, oneapi::mkl::rng::device::uniform_method::standard>>> test3;
    EXPECT_TRUEORSKIP((test3(GetParam())));
}

TEST_P(Mcg59UniformStdDeviceMomentsTests, UnsignedIntegerPrecision) {
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::mcg59<1>,
    oneapi::mkl::rng::device::uniform<std::uint32_t, oneapi::mkl::rng::device::uniform_method::standard>>> test1;
    EXPECT_TRUEORSKIP((test1(GetParam())));
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::mcg59<4>,
    oneapi::mkl::rng::device::uniform<std::uint32_t, oneapi::mkl::rng::device::uniform_method::standard>>> test2;
    EXPECT_TRUEORSKIP((test2(GetParam())));
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::mcg59<16>,
    oneapi::mkl::rng::device::uniform<std::uint32_t, oneapi::mkl::rng::device::uniform_method::standard>>> test3;
    EXPECT_TRUEORSKIP((test3(GetParam())));
}

TEST_P(Mcg59UniformAccDeviceMomentsTests, RealSinglePrecision) {
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::mcg59<1>,
    oneapi::mkl::rng::device::uniform<float, oneapi::mkl::rng::device::uniform_method::accurate>>> test1;
    EXPECT_TRUEORSKIP((test1(GetParam())));
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::mcg59<4>,
    oneapi::mkl::rng::device::uniform<float, oneapi::mkl::rng::device::uniform_method::accurate>>> test2;
    EXPECT_TRUEORSKIP((test2(GetParam())));
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::mcg59<16>,
    oneapi::mkl::rng::device::uniform<float, oneapi::mkl::rng::device::uniform_method::accurate>>> test3;
    EXPECT_TRUEORSKIP((test3(GetParam())));
}

TEST_P(Mcg59UniformAccDeviceMomentsTests, RealDoublePrecision) {
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::mcg59<1>,
    oneapi::mkl::rng::device::uniform<double, oneapi::mkl::rng::device::uniform_method::accurate>>> test1;
    EXPECT_TRUEORSKIP((test1(GetParam())));
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::mcg59<4>,
    oneapi::mkl::rng::device::uniform<double, oneapi::mkl::rng::device::uniform_method::accurate>>> test2;
    EXPECT_TRUEORSKIP((test2(GetParam())));
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::mcg59<16>,
    oneapi::mkl::rng::device::uniform<double, oneapi::mkl::rng::device::uniform_method::accurate>>> test3;
    EXPECT_TRUEORSKIP((test3(GetParam())));
}

TEST_P(Mcg59UniformAccDeviceMomentsTests, IntegerPrecision) {
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::mcg59<1>,
    oneapi::mkl::rng::device::uniform<std::int32_t, oneapi::mkl::rng::device::uniform_method::accurate>>> test1;
    EXPECT_TRUEORSKIP((test1(GetParam())));
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::mcg59<4>,
    oneapi::mkl::rng::device::uniform<std::int32_t, oneapi::mkl::rng::device::uniform_method::accurate>>> test2;
    EXPECT_TRUEORSKIP((test2(GetParam())));
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::mcg59<16>,
    oneapi::mkl::rng::device::uniform<std::int32_t, oneapi::mkl::rng::device::uniform_method::accurate>>> test3;
    EXPECT_TRUEORSKIP((test3(GetParam())));
}

TEST_P(Mcg59UniformAccDeviceMomentsTests, UnsignedIntegerPrecision) {
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::mcg59<1>,
    oneapi::mkl::rng::device::uniform<std::uint32_t, oneapi::mkl::rng::device::uniform_method::accurate>>> test1;
    EXPECT_TRUEORSKIP((test1(GetParam())));
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::mcg59<4>,
    oneapi::mkl::rng::device::uniform<std::uint32_t, oneapi::mkl::rng::device::uniform_method::accurate>>> test2;
    EXPECT_TRUEORSKIP((test2(GetParam())));
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::mcg59<16>,
    oneapi::mkl::rng::device::uniform<std::uint32_t, oneapi::mkl::rng::device::uniform_method::accurate>>> test3;
    EXPECT_TRUEORSKIP((test3(GetParam())));
}

INSTANTIATE_TEST_SUITE_P(Mcg59UniformStdDeviceMomentsTestsSuite, Mcg59UniformStdDeviceMomentsTests,
                         ::testing::ValuesIn(devices), ::DeviceNamePrint());

INSTANTIATE_TEST_SUITE_P(Mcg59UniformAccDeviceMomentsTestsSuite, Mcg59UniformAccDeviceMomentsTests,
                         ::testing::ValuesIn(devices), ::DeviceNamePrint());

class Philox4x32x10BitsDeviceMomentsTests : public ::testing::TestWithParam<sycl::device*> {};

TEST_P(Philox4x32x10BitsDeviceMomentsTests, UnsignedIntegerPrecision) {
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::philox4x32x10<1>,
    oneapi::mkl::rng::device::bits<uint32_t>>> test1;
    EXPECT_TRUEORSKIP((test1(GetParam())));
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::philox4x32x10<4>,
    oneapi::mkl::rng::device::bits<uint32_t>>> test2;
    EXPECT_TRUEORSKIP((test2(GetParam())));
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::philox4x32x10<16>,
    oneapi::mkl::rng::device::bits<uint32_t>>> test3;
    EXPECT_TRUEORSKIP((test3(GetParam())));
}

INSTANTIATE_TEST_SUITE_P(Philox4x32x10BitsDeviceMomentsTestsSuite, Philox4x32x10BitsDeviceMomentsTests,
                         ::testing::ValuesIn(devices), ::DeviceNamePrint());

class Philox4x32x10UniformBitsDeviceMomentsTests : public ::testing::TestWithParam<sycl::device*> {};

TEST_P(Philox4x32x10UniformBitsDeviceMomentsTests, UnsignedIntegerPrecision) {
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::philox4x32x10<1>,
    oneapi::mkl::rng::device::uniform_bits<uint32_t>>> test1;
    EXPECT_TRUEORSKIP((test1(GetParam())));
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::philox4x32x10<4>,
    oneapi::mkl::rng::device::uniform_bits<uint32_t>>> test2;
    EXPECT_TRUEORSKIP((test2(GetParam())));
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::philox4x32x10<16>,
    oneapi::mkl::rng::device::uniform_bits<uint32_t>>> test3;
    EXPECT_TRUEORSKIP((test3(GetParam())));
}

TEST_P(Philox4x32x10UniformBitsDeviceMomentsTests, UnsignedLongIntegerPrecision) {
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::philox4x32x10<1>,
    oneapi::mkl::rng::device::uniform_bits<uint64_t>>> test1;
    EXPECT_TRUEORSKIP((test1(GetParam())));
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::philox4x32x10<4>,
    oneapi::mkl::rng::device::uniform_bits<uint64_t>>> test2;
    EXPECT_TRUEORSKIP((test2(GetParam())));
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::philox4x32x10<16>,
    oneapi::mkl::rng::device::uniform_bits<uint64_t>>> test3;
    EXPECT_TRUEORSKIP((test3(GetParam())));
}

INSTANTIATE_TEST_SUITE_P(Philox4x32x10UniformBitsDeviceMomentsTestsSuite, Philox4x32x10UniformBitsDeviceMomentsTests,
                         ::testing::ValuesIn(devices), ::DeviceNamePrint());

class Philox4x32x10GaussianBoxMuller2DeviceMomentsTests : public ::testing::TestWithParam<sycl::device*> {};

TEST_P(Philox4x32x10GaussianBoxMuller2DeviceMomentsTests, RealSinglePrecision) {
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::philox4x32x10<1>,
    oneapi::mkl::rng::device::gaussian<float, oneapi::mkl::rng::device::gaussian_method::box_muller2>>> test1;
    EXPECT_TRUEORSKIP((test1(GetParam())));
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::philox4x32x10<4>,
    oneapi::mkl::rng::device::gaussian<float, oneapi::mkl::rng::device::gaussian_method::box_muller2>>> test2;
    EXPECT_TRUEORSKIP((test2(GetParam())));
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::philox4x32x10<16>,
    oneapi::mkl::rng::device::gaussian<float, oneapi::mkl::rng::device::gaussian_method::box_muller2>>> test3;
    EXPECT_TRUEORSKIP((test3(GetParam())));
}

TEST_P(Philox4x32x10GaussianBoxMuller2DeviceMomentsTests, RealDoublePrecision) {
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::philox4x32x10<1>,
    oneapi::mkl::rng::device::gaussian<double, oneapi::mkl::rng::device::gaussian_method::box_muller2>>> test1;
    EXPECT_TRUEORSKIP((test1(GetParam())));
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::philox4x32x10<4>,
    oneapi::mkl::rng::device::gaussian<double, oneapi::mkl::rng::device::gaussian_method::box_muller2>>> test2;
    EXPECT_TRUEORSKIP((test2(GetParam())));
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::philox4x32x10<16>,
    oneapi::mkl::rng::device::gaussian<double, oneapi::mkl::rng::device::gaussian_method::box_muller2>>> test3;
    EXPECT_TRUEORSKIP((test3(GetParam())));
}

INSTANTIATE_TEST_SUITE_P(Philox4x32x10GaussianBoxMuller2DeviceMomentsTestsSuite, Philox4x32x10GaussianBoxMuller2DeviceMomentsTests,
                         ::testing::ValuesIn(devices), ::DeviceNamePrint());

class Philox4x32x10LognormalBoxMuller2DeviceMomentsTests : public ::testing::TestWithParam<sycl::device*> {};

TEST_P(Philox4x32x10LognormalBoxMuller2DeviceMomentsTests, RealSinglePrecision) {
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::philox4x32x10<1>,
    oneapi::mkl::rng::device::lognormal<float, oneapi::mkl::rng::device::lognormal_method::box_muller2>>> test1;
    EXPECT_TRUEORSKIP((test1(GetParam())));
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::philox4x32x10<4>,
    oneapi::mkl::rng::device::lognormal<float, oneapi::mkl::rng::device::lognormal_method::box_muller2>>> test2;
    EXPECT_TRUEORSKIP((test2(GetParam())));
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::philox4x32x10<16>,
    oneapi::mkl::rng::device::lognormal<float, oneapi::mkl::rng::device::lognormal_method::box_muller2>>> test3;
    EXPECT_TRUEORSKIP((test3(GetParam())));
}

TEST_P(Philox4x32x10LognormalBoxMuller2DeviceMomentsTests, RealDoublePrecision) {
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::philox4x32x10<1>,
    oneapi::mkl::rng::device::lognormal<double, oneapi::mkl::rng::device::lognormal_method::box_muller2>>> test1;
    EXPECT_TRUEORSKIP((test1(GetParam())));
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::philox4x32x10<4>,
    oneapi::mkl::rng::device::lognormal<double, oneapi::mkl::rng::device::lognormal_method::box_muller2>>> test2;
    EXPECT_TRUEORSKIP((test2(GetParam())));
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::philox4x32x10<16>,
    oneapi::mkl::rng::device::lognormal<double, oneapi::mkl::rng::device::lognormal_method::box_muller2>>> test3;
    EXPECT_TRUEORSKIP((test3(GetParam())));
}

INSTANTIATE_TEST_SUITE_P(Philox4x32x10LognormalBoxMuller2DeviceMomentsTestsSuite, Philox4x32x10LognormalBoxMuller2DeviceMomentsTests,
                         ::testing::ValuesIn(devices), ::DeviceNamePrint());

class Philox4x32x10ExponentialIcdfDeviceMomentsTests : public ::testing::TestWithParam<sycl::device*> {};

class Philox4x32x10ExponentialIcdfAccDeviceMomentsTests : public ::testing::TestWithParam<sycl::device*> {};

TEST_P(Philox4x32x10ExponentialIcdfDeviceMomentsTests, RealSinglePrecision) {
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::philox4x32x10<1>,
    oneapi::mkl::rng::device::exponential<float, oneapi::mkl::rng::device::exponential_method::icdf>>> test1;
    EXPECT_TRUEORSKIP((test1(GetParam())));
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::philox4x32x10<4>,
    oneapi::mkl::rng::device::exponential<float, oneapi::mkl::rng::device::exponential_method::icdf>>> test2;
    EXPECT_TRUEORSKIP((test2(GetParam())));
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::philox4x32x10<16>,
    oneapi::mkl::rng::device::exponential<float, oneapi::mkl::rng::device::exponential_method::icdf>>> test3;
    EXPECT_TRUEORSKIP((test3(GetParam())));
}

TEST_P(Philox4x32x10ExponentialIcdfDeviceMomentsTests, RealDoublePrecision) {
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::philox4x32x10<1>,
    oneapi::mkl::rng::device::exponential<double, oneapi::mkl::rng::device::exponential_method::icdf>>> test1;
    EXPECT_TRUEORSKIP((test1(GetParam())));
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::philox4x32x10<4>,
    oneapi::mkl::rng::device::exponential<double, oneapi::mkl::rng::device::exponential_method::icdf>>> test2;
    EXPECT_TRUEORSKIP((test2(GetParam())));
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::philox4x32x10<16>,
    oneapi::mkl::rng::device::exponential<double, oneapi::mkl::rng::device::exponential_method::icdf>>> test3;
    EXPECT_TRUEORSKIP((test3(GetParam())));
}

TEST_P(Philox4x32x10ExponentialIcdfAccDeviceMomentsTests, RealSinglePrecision) {
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::philox4x32x10<1>,
    oneapi::mkl::rng::device::exponential<float, oneapi::mkl::rng::device::exponential_method::icdf_accurate>>> test1;
    EXPECT_TRUEORSKIP((test1(GetParam())));
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::philox4x32x10<4>,
    oneapi::mkl::rng::device::exponential<float, oneapi::mkl::rng::device::exponential_method::icdf_accurate>>> test2;
    EXPECT_TRUEORSKIP((test2(GetParam())));
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::philox4x32x10<16>,
    oneapi::mkl::rng::device::exponential<float, oneapi::mkl::rng::device::exponential_method::icdf_accurate>>> test3;
    EXPECT_TRUEORSKIP((test3(GetParam())));
}

TEST_P(Philox4x32x10ExponentialIcdfAccDeviceMomentsTests, RealDoublePrecision) {
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::philox4x32x10<1>,
    oneapi::mkl::rng::device::exponential<double, oneapi::mkl::rng::device::exponential_method::icdf_accurate>>> test1;
    EXPECT_TRUEORSKIP((test1(GetParam())));
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::philox4x32x10<4>,
    oneapi::mkl::rng::device::exponential<double, oneapi::mkl::rng::device::exponential_method::icdf_accurate>>> test2;
    EXPECT_TRUEORSKIP((test2(GetParam())));
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::philox4x32x10<16>,
    oneapi::mkl::rng::device::exponential<double, oneapi::mkl::rng::device::exponential_method::icdf_accurate>>> test3;
    EXPECT_TRUEORSKIP((test3(GetParam())));
}

INSTANTIATE_TEST_SUITE_P(Philox4x32x10ExponentialIcdfDeviceMomentsTestsSuite, Philox4x32x10ExponentialIcdfDeviceMomentsTests,
                         ::testing::ValuesIn(devices), ::DeviceNamePrint());

INSTANTIATE_TEST_SUITE_P(Philox4x32x10ExponentialIcdfAccDeviceMomentsTestsSuite, Philox4x32x10ExponentialIcdfAccDeviceMomentsTests,
                         ::testing::ValuesIn(devices), ::DeviceNamePrint());

class Philox4x32x10PoissonDevroyeDeviceMomentsTests : public ::testing::TestWithParam<sycl::device*> {};

TEST_P(Philox4x32x10PoissonDevroyeDeviceMomentsTests, IntegerPrecision) {
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::philox4x32x10<1>,
    oneapi::mkl::rng::device::poisson<int32_t, oneapi::mkl::rng::device::poisson_method::devroye>>> test1;
    EXPECT_TRUEORSKIP((test1(GetParam())));
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::philox4x32x10<4>,
    oneapi::mkl::rng::device::poisson<int32_t, oneapi::mkl::rng::device::poisson_method::devroye>>> test2;
    EXPECT_TRUEORSKIP((test2(GetParam())));
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::philox4x32x10<16>,
    oneapi::mkl::rng::device::poisson<int32_t, oneapi::mkl::rng::device::poisson_method::devroye>>> test3;
    EXPECT_TRUEORSKIP((test3(GetParam())));
}

TEST_P(Philox4x32x10PoissonDevroyeDeviceMomentsTests, UnsignedIntegerPrecision) {
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::philox4x32x10<1>,
    oneapi::mkl::rng::device::poisson<uint32_t, oneapi::mkl::rng::device::poisson_method::devroye>>> test1;
    EXPECT_TRUEORSKIP((test1(GetParam())));
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::philox4x32x10<4>,
    oneapi::mkl::rng::device::poisson<uint32_t, oneapi::mkl::rng::device::poisson_method::devroye>>> test2;
    EXPECT_TRUEORSKIP((test2(GetParam())));
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::philox4x32x10<16>,
    oneapi::mkl::rng::device::poisson<uint32_t, oneapi::mkl::rng::device::poisson_method::devroye>>> test3;
    EXPECT_TRUEORSKIP((test3(GetParam())));
}

INSTANTIATE_TEST_SUITE_P(Philox4x32x10PoissonDevroyeDeviceMomentsTestsSuite, Philox4x32x10PoissonDevroyeDeviceMomentsTests,
                         ::testing::ValuesIn(devices), ::DeviceNamePrint());

class Philox4x32x10BernoulliIcdfDeviceMomentsTests : public ::testing::TestWithParam<sycl::device*> {};

TEST_P(Philox4x32x10BernoulliIcdfDeviceMomentsTests, IntegerPrecision) {
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::philox4x32x10<1>,
    oneapi::mkl::rng::device::bernoulli<int32_t, oneapi::mkl::rng::device::bernoulli_method::icdf>>> test1;
    EXPECT_TRUEORSKIP((test1(GetParam())));
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::philox4x32x10<4>,
    oneapi::mkl::rng::device::bernoulli<int32_t, oneapi::mkl::rng::device::bernoulli_method::icdf>>> test2;
    EXPECT_TRUEORSKIP((test2(GetParam())));
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::philox4x32x10<16>,
    oneapi::mkl::rng::device::bernoulli<int32_t, oneapi::mkl::rng::device::bernoulli_method::icdf>>> test3;
    EXPECT_TRUEORSKIP((test3(GetParam())));
}

TEST_P(Philox4x32x10BernoulliIcdfDeviceMomentsTests, UnsignedIntegerPrecision) {
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::philox4x32x10<1>,
    oneapi::mkl::rng::device::bernoulli<uint32_t, oneapi::mkl::rng::device::bernoulli_method::icdf>>> test1;
    EXPECT_TRUEORSKIP((test1(GetParam())));
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::philox4x32x10<4>,
    oneapi::mkl::rng::device::bernoulli<uint32_t, oneapi::mkl::rng::device::bernoulli_method::icdf>>> test2;
    EXPECT_TRUEORSKIP((test2(GetParam())));
    rng_device_test<moments_test<
    oneapi::mkl::rng::device::philox4x32x10<16>,
    oneapi::mkl::rng::device::bernoulli<uint32_t, oneapi::mkl::rng::device::bernoulli_method::icdf>>> test3;
    EXPECT_TRUEORSKIP((test3(GetParam())));
}

INSTANTIATE_TEST_SUITE_P(Philox4x32x10BernoulliIcdfDeviceMomentsTestsSuite, Philox4x32x10BernoulliIcdfDeviceMomentsTests,
                         ::testing::ValuesIn(devices), ::DeviceNamePrint());

} // anonymous namespace
