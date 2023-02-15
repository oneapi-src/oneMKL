/***************************************************************************
*  Copyright (C) Codeplay Software Limited
*  Licensed under the Apache License, Version 2.0 (the "License");
*  you may not use this file except in compliance with the License.
*  You may obtain a copy of the License at
*
*      http://www.apache.org/licenses/LICENSE-2.0
*
*  For your convenience, a copy of the License has been included in this
*  repository.
*
*  Unless required by applicable law or agreed to in writing, software
*  distributed under the License is distributed on an "AS IS" BASIS,
*  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*  See the License for the specific language governing permissions and
*  limitations under the License.
*
**************************************************************************/

#include <iostream>
#include <vector>

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include "test_helper.hpp"
#include "test_common.hpp"
#include <gtest/gtest.h>

#include "compute_inplace.hpp"
#include "compute_inplace_real_real.hpp"
#include "compute_out_of_place.hpp"
#include "compute_out_of_place_real_real.hpp"

extern std::vector<sycl::device *> devices;

namespace {

class ComputeTests : public ::testing::TestWithParam<std::tuple<sycl::device *, std::int64_t>> {};

std::vector<std::int64_t> lengths{ 8, 21, 128 };

/* test_in_place_buffer() */
TEST_P(ComputeTests, RealSinglePrecisionInPlaceBuffer) {
    auto test = DFT_Test<oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::REAL>{
        std::get<0>(GetParam()), std::get<1>(GetParam())
    };
    EXPECT_TRUEORSKIP(test.test_in_place_buffer());
}

TEST_P(ComputeTests, RealDoublePrecisionInPlaceBuffer) {
    auto test = DFT_Test<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>{
        std::get<0>(GetParam()), std::get<1>(GetParam())
    };
    EXPECT_TRUEORSKIP(test.test_in_place_buffer());
}

TEST_P(ComputeTests, ComplexSinglePrecisionInPlaceBuffer) {
    auto test = DFT_Test<oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::COMPLEX>{
        std::get<0>(GetParam()), std::get<1>(GetParam())
    };
    EXPECT_TRUEORSKIP(test.test_in_place_buffer());
}

TEST_P(ComputeTests, ComplexDoublePrecisionInPlaceBuffer) {
    auto test = DFT_Test<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::COMPLEX>{
        std::get<0>(GetParam()), std::get<1>(GetParam())
    };
    EXPECT_TRUEORSKIP(test.test_in_place_buffer());
}

/* test_in_place_real_real_buffer() */
TEST_P(ComputeTests, RealSinglePrecisionInPlaceRealRealBuffer) {
    auto test = DFT_Test<oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::REAL>{
        std::get<0>(GetParam()), std::get<1>(GetParam())
    };
    EXPECT_TRUEORSKIP(test.test_in_place_real_real_buffer());
}

TEST_P(ComputeTests, RealDoublePrecisionInPlaceRealRealBuffer) {
    auto test = DFT_Test<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>{
        std::get<0>(GetParam()), std::get<1>(GetParam())
    };
    EXPECT_TRUEORSKIP(test.test_in_place_real_real_buffer());
}

TEST_P(ComputeTests, ComplexSinglePrecisionInPlaceRealRealBuffer) {
    auto test = DFT_Test<oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::COMPLEX>{
        std::get<0>(GetParam()), std::get<1>(GetParam())
    };
    EXPECT_TRUEORSKIP(test.test_in_place_real_real_buffer());
}

TEST_P(ComputeTests, ComplexDoublePrecisionInPlaceRealRealBuffer) {
    auto test = DFT_Test<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::COMPLEX>{
        std::get<0>(GetParam()), std::get<1>(GetParam())
    };
    EXPECT_TRUEORSKIP(test.test_in_place_real_real_buffer());
}

/* test_out_of_place_buffer() */
TEST_P(ComputeTests, RealSinglePrecisionNotInPlaceBuffer) {
    auto test = DFT_Test<oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::REAL>{
        std::get<0>(GetParam()), std::get<1>(GetParam())
    };
    EXPECT_TRUEORSKIP(test.test_out_of_place_buffer());
}

TEST_P(ComputeTests, RealDoublePrecisionNotInPlaceBuffer) {
    auto test = DFT_Test<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>{
        std::get<0>(GetParam()), std::get<1>(GetParam())
    };
    EXPECT_TRUEORSKIP(test.test_out_of_place_buffer());
}

TEST_P(ComputeTests, ComplexSinglePrecisionNotInPlaceBuffer) {
    auto test = DFT_Test<oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::COMPLEX>{
        std::get<0>(GetParam()), std::get<1>(GetParam())
    };
    EXPECT_TRUEORSKIP(test.test_out_of_place_buffer());
}

TEST_P(ComputeTests, ComplexDoublePrecisionNotInPlaceBuffer) {
    auto test = DFT_Test<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::COMPLEX>{
        std::get<0>(GetParam()), std::get<1>(GetParam())
    };
    EXPECT_TRUEORSKIP(test.test_out_of_place_buffer());
}

/* test_out_of_place_real_real_buffer */
TEST_P(ComputeTests, RealSinglePrecisionNotInPlaceRealRealBuffer) {
    auto test = DFT_Test<oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::REAL>{
        std::get<0>(GetParam()), std::get<1>(GetParam())
    };
    EXPECT_TRUEORSKIP(test.test_out_of_place_real_real_buffer());
}

TEST_P(ComputeTests, RealDoublePrecisionNotInPlaceRealRealBuffer) {
    auto test = DFT_Test<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>{
        std::get<0>(GetParam()), std::get<1>(GetParam())
    };
    EXPECT_TRUEORSKIP(test.test_out_of_place_real_real_buffer());
}

TEST_P(ComputeTests, ComplexSinglePrecisionNotInPlaceRealRealBuffer) {
    auto test = DFT_Test<oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::COMPLEX>{
        std::get<0>(GetParam()), std::get<1>(GetParam())
    };
    EXPECT_TRUEORSKIP(test.test_out_of_place_real_real_buffer());
}

TEST_P(ComputeTests, ComplexDoublePrecisionNotInPlaceRealRealBuffer) {
    auto test = DFT_Test<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::COMPLEX>{
        std::get<0>(GetParam()), std::get<1>(GetParam())
    };
    EXPECT_TRUEORSKIP(test.test_out_of_place_real_real_buffer());
}

/* test_in_place_USM */
TEST_P(ComputeTests, RealSinglePrecisionInPlaceUSM) {
    auto test = DFT_Test<oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::REAL>{
        std::get<0>(GetParam()), std::get<1>(GetParam())
    };
    EXPECT_TRUEORSKIP(test.test_in_place_USM());
}

TEST_P(ComputeTests, RealDoublePrecisionInPlaceUSM) {
    auto test = DFT_Test<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>{
        std::get<0>(GetParam()), std::get<1>(GetParam())
    };
    EXPECT_TRUEORSKIP(test.test_in_place_USM());
}

TEST_P(ComputeTests, ComplexSinglePrecisionInPlaceUSM) {
    auto test = DFT_Test<oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::COMPLEX>{
        std::get<0>(GetParam()), std::get<1>(GetParam())
    };
    EXPECT_TRUEORSKIP(test.test_in_place_USM());
}

TEST_P(ComputeTests, ComplexDoublePrecisionInPlaceUSM) {
    auto test = DFT_Test<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::COMPLEX>{
        std::get<0>(GetParam()), std::get<1>(GetParam())
    };
    EXPECT_TRUEORSKIP(test.test_in_place_USM());
}

/* test_in_place_real_real_USM */
TEST_P(ComputeTests, RealSinglePrecisionInPlaceRealRealUSM) {
    auto test = DFT_Test<oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::REAL>{
        std::get<0>(GetParam()), std::get<1>(GetParam())
    };
    EXPECT_TRUEORSKIP(test.test_in_place_real_real_USM());
}

TEST_P(ComputeTests, RealDoublePrecisionInPlaceRealRealUSM) {
    auto test = DFT_Test<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>{
        std::get<0>(GetParam()), std::get<1>(GetParam())
    };
    EXPECT_TRUEORSKIP(test.test_in_place_real_real_USM());
}

TEST_P(ComputeTests, ComplexSinglePrecisionInPlaceRealRealUSM) {
    auto test = DFT_Test<oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::COMPLEX>{
        std::get<0>(GetParam()), std::get<1>(GetParam())
    };
    EXPECT_TRUEORSKIP(test.test_in_place_real_real_USM());
}

TEST_P(ComputeTests, ComplexDoublePrecisionInPlaceRealRealUSM) {
    auto test = DFT_Test<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::COMPLEX>{
        std::get<0>(GetParam()), std::get<1>(GetParam())
    };
    EXPECT_TRUEORSKIP(test.test_in_place_real_real_USM());
}

/* test_out_of_place_USM */
TEST_P(ComputeTests, RealSinglePrecisionNotInPlaceUSM) {
    auto test = DFT_Test<oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::REAL>{
        std::get<0>(GetParam()), std::get<1>(GetParam())
    };
    EXPECT_TRUEORSKIP(test.test_out_of_place_USM());
}

TEST_P(ComputeTests, RealDoublePrecisionNotInPlaceUSM) {
    auto test = DFT_Test<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>{
        std::get<0>(GetParam()), std::get<1>(GetParam())
    };
    EXPECT_TRUEORSKIP(test.test_out_of_place_USM());
}

TEST_P(ComputeTests, ComplexSinglePrecisionNotInPlaceUSM) {
    auto test = DFT_Test<oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::COMPLEX>{
        std::get<0>(GetParam()), std::get<1>(GetParam())
    };
    EXPECT_TRUEORSKIP(test.test_out_of_place_USM());
}

TEST_P(ComputeTests, ComplexDoublePrecisionNotInPlaceUSM) {
    auto test = DFT_Test<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::COMPLEX>{
        std::get<0>(GetParam()), std::get<1>(GetParam())
    };
    EXPECT_TRUEORSKIP(test.test_out_of_place_USM());
}

/* test_out_of_place_real_real_USM */
TEST_P(ComputeTests, RealSinglePrecisionNotInPlaceRealRealUSM) {
    auto test = DFT_Test<oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::REAL>{
        std::get<0>(GetParam()), std::get<1>(GetParam())
    };
    EXPECT_TRUEORSKIP(test.test_out_of_place_real_real_USM());
}

TEST_P(ComputeTests, RealDoublePrecisionNotInPlaceRealRealUSM) {
    auto test = DFT_Test<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>{
        std::get<0>(GetParam()), std::get<1>(GetParam())
    };
    EXPECT_TRUEORSKIP(test.test_out_of_place_real_real_USM());
}

TEST_P(ComputeTests, ComplexSinglePrecisionNotInPlaceRealRealUSM) {
    auto test = DFT_Test<oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::COMPLEX>{
        std::get<0>(GetParam()), std::get<1>(GetParam())
    };
    EXPECT_TRUEORSKIP(test.test_out_of_place_real_real_USM());
}

TEST_P(ComputeTests, ComplexDoublePrecisionNotInPlaceRealRealUSM) {
    auto test = DFT_Test<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::COMPLEX>{
        std::get<0>(GetParam()), std::get<1>(GetParam())
    };
    EXPECT_TRUEORSKIP(test.test_out_of_place_real_real_USM());
}

INSTANTIATE_TEST_SUITE_P(ComputeTestSuite, ComputeTests,
                         ::testing::Combine(testing::ValuesIn(devices), testing::ValuesIn(lengths)),
                         ::DimensionsDeviceNamePrint());

} // anonymous namespace
