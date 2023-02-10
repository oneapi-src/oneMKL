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

extern std::vector<sycl::device*> devices;

namespace {

constexpr std::int64_t default_1d_lengths = 4;
const std::vector<std::int64_t> default_3d_lengths{ 124, 5, 3 };

template <oneapi::mkl::dft::precision precision, oneapi::mkl::dft::domain domain>
inline void set_and_get_lengths(sycl::queue& sycl_queue) {
    /* Negative Testing */
    {
        oneapi::mkl::dft::descriptor<precision, domain> descriptor{ default_3d_lengths };
        EXPECT_THROW(descriptor.set_value(oneapi::mkl::dft::config_param::LENGTHS, nullptr),
                     oneapi::mkl::invalid_argument);
    }

    /* 1D */
    {
        oneapi::mkl::dft::descriptor<precision, domain> descriptor{ default_1d_lengths };

        std::int64_t lengths_value{ 0 };
        std::int64_t new_lengths{ 2345 };
        std::int64_t dimensions_before_set{ 0 };
        std::int64_t dimensions_after_set{ 0 };

        descriptor.get_value(oneapi::mkl::dft::config_param::LENGTHS, &lengths_value);
        descriptor.get_value(oneapi::mkl::dft::config_param::DIMENSION, &dimensions_before_set);
        EXPECT_EQ(default_1d_lengths, lengths_value);
        EXPECT_EQ(dimensions_before_set, 1);

        descriptor.set_value(oneapi::mkl::dft::config_param::LENGTHS, new_lengths);
        descriptor.get_value(oneapi::mkl::dft::config_param::LENGTHS, &lengths_value);
        descriptor.get_value(oneapi::mkl::dft::config_param::DIMENSION, &dimensions_after_set);
        EXPECT_EQ(new_lengths, lengths_value);
        EXPECT_EQ(dimensions_before_set, dimensions_after_set);

        commit_descriptor(descriptor, sycl_queue);
    }

    /* >= 2D */
    {
        const std::int64_t dimensions = 3;

        oneapi::mkl::dft::descriptor<precision, domain> descriptor{ default_3d_lengths };

        std::vector<std::int64_t> lengths_value(3);
        std::vector<std::int64_t> new_lengths{ 1, 2, 7 };
        std::int64_t dimensions_before_set{ 0 };
        std::int64_t dimensions_after_set{ 0 };

        descriptor.get_value(oneapi::mkl::dft::config_param::LENGTHS, lengths_value.data());
        descriptor.get_value(oneapi::mkl::dft::config_param::DIMENSION, &dimensions_before_set);

        EXPECT_EQ(default_3d_lengths, lengths_value);
        EXPECT_EQ(dimensions, dimensions_before_set);

        descriptor.set_value(oneapi::mkl::dft::config_param::LENGTHS, new_lengths.data());
        descriptor.get_value(oneapi::mkl::dft::config_param::LENGTHS, lengths_value.data());
        descriptor.get_value(oneapi::mkl::dft::config_param::DIMENSION, &dimensions_after_set);

        EXPECT_EQ(new_lengths, lengths_value);
        EXPECT_EQ(dimensions_before_set, dimensions_after_set);
    }
}

template <oneapi::mkl::dft::precision precision, oneapi::mkl::dft::domain domain>
inline void set_and_get_strides(sycl::queue& sycl_queue) {
    oneapi::mkl::dft::descriptor<precision, domain> descriptor{ default_3d_lengths };

    EXPECT_THROW(descriptor.set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, nullptr),
                 oneapi::mkl::invalid_argument);
    EXPECT_THROW(descriptor.set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, nullptr),
                 oneapi::mkl::invalid_argument);

    constexpr std::int64_t strides_size = 4;
    const std::int64_t default_stride_d1 = default_3d_lengths[2] * default_3d_lengths[1];
    const std::int64_t default_stride_d2 = default_3d_lengths[2];
    const std::int64_t default_stride_d3 = 1;

    std::vector<std::int64_t> default_strides_value{ 0, default_stride_d1, default_stride_d2,
                                                     default_stride_d3 };

    std::vector<std::int64_t> input_strides_value;
    std::vector<std::int64_t> output_strides_value;
    if constexpr (domain == oneapi::mkl::dft::domain::COMPLEX) {
        input_strides_value = { 50, default_stride_d1 * 2, default_stride_d2 * 2,
                                default_stride_d3 * 2 };
        output_strides_value = { 50, default_stride_d1 * 2, default_stride_d2 * 2,
                                 default_stride_d3 * 2 };
    }
    else {
        input_strides_value = { 0, default_3d_lengths[1] * (default_3d_lengths[2] / 2 + 1) * 2,
                                (default_3d_lengths[2] / 2 + 1) * 2, 1 };
        output_strides_value = { 0, default_3d_lengths[1] * (default_3d_lengths[2] / 2 + 1),
                                 (default_3d_lengths[2] / 2 + 1), 1 };
    }

    std::vector<std::int64_t> input_strides_before_set(strides_size);
    std::vector<std::int64_t> input_strides_after_set(strides_size);

    descriptor.get_value(oneapi::mkl::dft::config_param::INPUT_STRIDES,
                         input_strides_before_set.data());
    EXPECT_EQ(default_strides_value, input_strides_before_set);
    descriptor.set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, input_strides_value.data());
    descriptor.get_value(oneapi::mkl::dft::config_param::INPUT_STRIDES,
                         input_strides_after_set.data());
    EXPECT_EQ(input_strides_value, input_strides_after_set);

    std::vector<std::int64_t> output_strides_before_set(strides_size);
    std::vector<std::int64_t> output_strides_after_set(strides_size);
    descriptor.get_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES,
                         output_strides_before_set.data());
    EXPECT_EQ(default_strides_value, output_strides_before_set);
    descriptor.set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES,
                         output_strides_value.data());
    descriptor.get_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES,
                         output_strides_after_set.data());
    EXPECT_EQ(output_strides_value, output_strides_after_set);
}

template <oneapi::mkl::dft::precision precision, oneapi::mkl::dft::domain domain>
inline void set_and_get_values(sycl::queue& sycl_queue) {
    oneapi::mkl::dft::descriptor<precision, domain> descriptor{ default_1d_lengths };

    using Precision_Type =
        typename std::conditional_t<precision == oneapi::mkl::dft::precision::SINGLE, float,
                                    double>;

    {
        Precision_Type forward_scale_set_value{ 143.5 };
        Precision_Type forward_scale_before_set;
        Precision_Type forward_scale_after_set;

        descriptor.get_value(oneapi::mkl::dft::config_param::FORWARD_SCALE,
                             &forward_scale_before_set);
        EXPECT_EQ(1.0, forward_scale_before_set);
        descriptor.set_value(oneapi::mkl::dft::config_param::FORWARD_SCALE,
                             forward_scale_set_value);
        descriptor.get_value(oneapi::mkl::dft::config_param::FORWARD_SCALE,
                             &forward_scale_after_set);
        EXPECT_EQ(forward_scale_set_value, forward_scale_after_set);
    }

    {
        Precision_Type backward_scale_set_value{ 143.5 };
        Precision_Type backward_scale_before_set;
        Precision_Type backward_scale_after_set;

        descriptor.get_value(oneapi::mkl::dft::config_param::BACKWARD_SCALE,
                             &backward_scale_before_set);
        EXPECT_EQ(1.0, backward_scale_before_set);
        descriptor.set_value(oneapi::mkl::dft::config_param::BACKWARD_SCALE,
                             backward_scale_set_value);
        descriptor.get_value(oneapi::mkl::dft::config_param::BACKWARD_SCALE,
                             &backward_scale_after_set);
        EXPECT_EQ(backward_scale_set_value, backward_scale_after_set);
    }

    {
        std::int64_t n_transforms_set_value{ 12 };
        std::int64_t n_transforms_before_set;
        std::int64_t n_transforms_after_set;

        descriptor.get_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS,
                             &n_transforms_before_set);
        EXPECT_EQ(1, n_transforms_before_set);
        descriptor.set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS,
                             n_transforms_set_value);
        descriptor.get_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS,
                             &n_transforms_after_set);
        EXPECT_EQ(n_transforms_set_value, n_transforms_after_set);
    }

    {
        std::int64_t fwd_distance_set_value{ 12 };
        std::int64_t fwd_distance_before_set;
        std::int64_t fwd_distance_after_set;

        descriptor.get_value(oneapi::mkl::dft::config_param::FWD_DISTANCE,
                             &fwd_distance_before_set);
        EXPECT_EQ(1, fwd_distance_before_set);
        descriptor.set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, fwd_distance_set_value);
        descriptor.get_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, &fwd_distance_after_set);
        EXPECT_EQ(fwd_distance_set_value, fwd_distance_after_set);

        std::int64_t bwd_distance_set_value{ domain == oneapi::mkl::dft::domain::REAL
                                                 ? (fwd_distance_set_value / 2) + 1
                                                 : fwd_distance_set_value };
        std::int64_t bwd_distance_before_set;
        std::int64_t bwd_distance_after_set;

        descriptor.get_value(oneapi::mkl::dft::config_param::BWD_DISTANCE,
                             &bwd_distance_before_set);
        EXPECT_EQ(1, bwd_distance_before_set);
        descriptor.set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, bwd_distance_set_value);
        descriptor.get_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, &bwd_distance_after_set);
        EXPECT_EQ(bwd_distance_set_value, bwd_distance_after_set);
    }

    {
        oneapi::mkl::dft::config_value value{
            oneapi::mkl::dft::config_value::COMMITTED
        }; // Initialize with invalid value
        descriptor.get_value(oneapi::mkl::dft::config_param::PLACEMENT, &value);
        EXPECT_EQ(oneapi::mkl::dft::config_value::INPLACE, value);

        descriptor.set_value(oneapi::mkl::dft::config_param::PLACEMENT,
                             oneapi::mkl::dft::config_value::NOT_INPLACE);
        descriptor.get_value(oneapi::mkl::dft::config_param::PLACEMENT, &value);
        EXPECT_EQ(oneapi::mkl::dft::config_value::NOT_INPLACE, value);

        descriptor.set_value(oneapi::mkl::dft::config_param::PLACEMENT,
                             oneapi::mkl::dft::config_value::INPLACE);
        descriptor.get_value(oneapi::mkl::dft::config_param::PLACEMENT, &value);
        EXPECT_EQ(oneapi::mkl::dft::config_value::INPLACE, value);
    }

    {
        oneapi::mkl::dft::config_value value{
            oneapi::mkl::dft::config_value::COMMITTED
        }; // Initialize with invalid value
        descriptor.get_value(oneapi::mkl::dft::config_param::COMPLEX_STORAGE, &value);
        EXPECT_EQ(oneapi::mkl::dft::config_value::COMPLEX_COMPLEX, value);

        descriptor.set_value(oneapi::mkl::dft::config_param::COMPLEX_STORAGE,
                             oneapi::mkl::dft::config_value::REAL_REAL);
        descriptor.get_value(oneapi::mkl::dft::config_param::COMPLEX_STORAGE, &value);
        EXPECT_EQ(oneapi::mkl::dft::config_value::REAL_REAL, value);

        descriptor.set_value(oneapi::mkl::dft::config_param::COMPLEX_STORAGE,
                             oneapi::mkl::dft::config_value::COMPLEX_COMPLEX);
        descriptor.get_value(oneapi::mkl::dft::config_param::COMPLEX_STORAGE, &value);
        EXPECT_EQ(oneapi::mkl::dft::config_value::COMPLEX_COMPLEX, value);
    }

    {
        oneapi::mkl::dft::config_value value{
            oneapi::mkl::dft::config_value::COMMITTED
        }; // Initialize with invalid value
        descriptor.get_value(oneapi::mkl::dft::config_param::CONJUGATE_EVEN_STORAGE, &value);
        EXPECT_EQ(oneapi::mkl::dft::config_value::COMPLEX_COMPLEX, value);

        descriptor.set_value(oneapi::mkl::dft::config_param::CONJUGATE_EVEN_STORAGE,
                             oneapi::mkl::dft::config_value::COMPLEX_COMPLEX);

        value = oneapi::mkl::dft::config_value::COMMITTED; // Initialize with invalid value
        descriptor.get_value(oneapi::mkl::dft::config_param::CONJUGATE_EVEN_STORAGE, &value);
        EXPECT_EQ(oneapi::mkl::dft::config_value::COMPLEX_COMPLEX, value);
    }

    {
        oneapi::mkl::dft::config_value value{
            oneapi::mkl::dft::config_value::COMMITTED
        }; // Initialize with invalid value
        descriptor.get_value(oneapi::mkl::dft::config_param::REAL_STORAGE, &value);
        EXPECT_EQ(oneapi::mkl::dft::config_value::REAL_REAL, value);

        descriptor.set_value(oneapi::mkl::dft::config_param::REAL_STORAGE,
                             oneapi::mkl::dft::config_value::REAL_REAL);

        value = oneapi::mkl::dft::config_value::COMMITTED; // Initialize with invalid value
        descriptor.get_value(oneapi::mkl::dft::config_param::REAL_STORAGE, &value);
        EXPECT_EQ(oneapi::mkl::dft::config_value::REAL_REAL, value);
    }

    {
        oneapi::mkl::dft::config_value value{
            oneapi::mkl::dft::config_value::COMMITTED
        }; // Initialize with invalid value
        descriptor.get_value(oneapi::mkl::dft::config_param::ORDERING, &value);
        EXPECT_EQ(oneapi::mkl::dft::config_value::ORDERED, value);

        descriptor.set_value(oneapi::mkl::dft::config_param::ORDERING,
                             oneapi::mkl::dft::config_value::BACKWARD_SCRAMBLED);
        descriptor.get_value(oneapi::mkl::dft::config_param::ORDERING, &value);
        EXPECT_EQ(oneapi::mkl::dft::config_value::BACKWARD_SCRAMBLED, value);

        descriptor.set_value(oneapi::mkl::dft::config_param::ORDERING,
                             oneapi::mkl::dft::config_value::ORDERED);
        descriptor.get_value(oneapi::mkl::dft::config_param::ORDERING, &value);
        EXPECT_EQ(oneapi::mkl::dft::config_value::ORDERED, value);
    }

    {
        bool value = true;
        descriptor.get_value(oneapi::mkl::dft::config_param::TRANSPOSE, &value);
        EXPECT_EQ(false, value);

        descriptor.set_value(oneapi::mkl::dft::config_param::TRANSPOSE, true);
        descriptor.get_value(oneapi::mkl::dft::config_param::TRANSPOSE, &value);
        EXPECT_EQ(true, value);
        /* Set value to false again because transpose is not implemented and will fail on commit
         * when using the MKLGPU backend */
        descriptor.set_value(oneapi::mkl::dft::config_param::TRANSPOSE, false);
    }

    {
        /* Only value currently supported for PACKED_FORMAT is the config_value::CCE_FORMAT */
        oneapi::mkl::dft::config_value value{
            oneapi::mkl::dft::config_value::COMMITTED
        }; // Initialize with invalid value
        descriptor.get_value(oneapi::mkl::dft::config_param::PACKED_FORMAT, &value);
        EXPECT_EQ(oneapi::mkl::dft::config_value::CCE_FORMAT, value);

        descriptor.set_value(oneapi::mkl::dft::config_param::PACKED_FORMAT,
                             oneapi::mkl::dft::config_value::CCE_FORMAT);

        value = oneapi::mkl::dft::config_value::COMMITTED; // Initialize with invalid value
        descriptor.get_value(oneapi::mkl::dft::config_param::PACKED_FORMAT, &value);
        EXPECT_EQ(oneapi::mkl::dft::config_value::CCE_FORMAT, value);
    }
}

template <oneapi::mkl::dft::precision precision, oneapi::mkl::dft::domain domain>
inline void get_readonly_values(sycl::queue& sycl_queue) {
    oneapi::mkl::dft::descriptor<precision, domain> descriptor{ default_1d_lengths };

    oneapi::mkl::dft::domain domain_value;
    descriptor.get_value(oneapi::mkl::dft::config_param::FORWARD_DOMAIN, &domain_value);
    EXPECT_EQ(domain_value, domain);

    oneapi::mkl::dft::precision precision_value;
    descriptor.get_value(oneapi::mkl::dft::config_param::PRECISION, &precision_value);
    EXPECT_EQ(precision_value, precision);

    std::int64_t dimension_value;
    descriptor.get_value(oneapi::mkl::dft::config_param::DIMENSION, &dimension_value);
    EXPECT_EQ(dimension_value, 1);

    oneapi::mkl::dft::descriptor<precision, domain> descriptor3D{ default_3d_lengths };
    descriptor3D.get_value(oneapi::mkl::dft::config_param::DIMENSION, &dimension_value);
    EXPECT_EQ(dimension_value, 3);

    oneapi::mkl::dft::config_value commit_status;
    descriptor.get_value(oneapi::mkl::dft::config_param::COMMIT_STATUS, &commit_status);
    EXPECT_EQ(commit_status, oneapi::mkl::dft::config_value::UNCOMMITTED);

    commit_descriptor(descriptor, sycl_queue);
    descriptor.get_value(oneapi::mkl::dft::config_param::COMMIT_STATUS, &commit_status);
    EXPECT_EQ(commit_status, oneapi::mkl::dft::config_value::COMMITTED);
}

template <oneapi::mkl::dft::precision precision, oneapi::mkl::dft::domain domain>
inline void set_readonly_values(sycl::queue& sycl_queue) {
    oneapi::mkl::dft::descriptor<precision, domain> descriptor{ default_1d_lengths };

    EXPECT_THROW(descriptor.set_value(oneapi::mkl::dft::config_param::FORWARD_DOMAIN,
                                      oneapi::mkl::dft::domain::REAL),
                 oneapi::mkl::invalid_argument);
    EXPECT_THROW(descriptor.set_value(oneapi::mkl::dft::config_param::FORWARD_DOMAIN,
                                      oneapi::mkl::dft::domain::COMPLEX),
                 oneapi::mkl::invalid_argument);

    EXPECT_THROW(descriptor.set_value(oneapi::mkl::dft::config_param::PRECISION,
                                      oneapi::mkl::dft::precision::SINGLE),
                 oneapi::mkl::invalid_argument);
    EXPECT_THROW(descriptor.set_value(oneapi::mkl::dft::config_param::PRECISION,
                                      oneapi::mkl::dft::precision::DOUBLE),
                 oneapi::mkl::invalid_argument);

    std::int64_t set_dimension{ 3 };
    EXPECT_THROW(descriptor.set_value(oneapi::mkl::dft::config_param::DIMENSION, set_dimension),
                 oneapi::mkl::invalid_argument);

    EXPECT_THROW(descriptor.set_value(oneapi::mkl::dft::config_param::COMMIT_STATUS,
                                      oneapi::mkl::dft::config_value::COMMITTED),
                 oneapi::mkl::invalid_argument);
    EXPECT_THROW(descriptor.set_value(oneapi::mkl::dft::config_param::COMMIT_STATUS,
                                      oneapi::mkl::dft::config_value::UNCOMMITTED),
                 oneapi::mkl::invalid_argument);

    commit_descriptor(descriptor, sycl_queue);
}

template <oneapi::mkl::dft::precision precision, oneapi::mkl::dft::domain domain>
int test(sycl::device* dev) {
    sycl::queue sycl_queue(*dev, exception_handler);

    if constexpr (precision == oneapi::mkl::dft::precision::DOUBLE) {
        if (!sycl_queue.get_device().has(sycl::aspect::fp64)) {
            std::cout << "Device does not support double precision." << std::endl;
            return test_skipped;
        }
    }

    set_and_get_lengths<precision, domain>(sycl_queue);
    set_and_get_strides<precision, domain>(sycl_queue);
    set_and_get_values<precision, domain>(sycl_queue);
    get_readonly_values<precision, domain>(sycl_queue);
    set_readonly_values<precision, domain>(sycl_queue);

    return !::testing::Test::HasFailure();
}

class DescriptorTests : public ::testing::TestWithParam<sycl::device*> {};

TEST_P(DescriptorTests, DescriptorTestsRealSingle) {
    EXPECT_TRUEORSKIP(
        (test<oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::REAL>(GetParam())));
}

TEST_P(DescriptorTests, DescriptorTestsRealDouble) {
    EXPECT_TRUEORSKIP(
        (test<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::REAL>(GetParam())));
}

TEST_P(DescriptorTests, DescriptorTestsComplexSingle) {
    EXPECT_TRUEORSKIP(
        (test<oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::COMPLEX>(GetParam())));
}

TEST_P(DescriptorTests, DescriptorTestsComplexDouble) {
    EXPECT_TRUEORSKIP(
        (test<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::COMPLEX>(GetParam())));
}

INSTANTIATE_TEST_SUITE_P(DescriptorTestSuite, DescriptorTests, testing::ValuesIn(devices),
                         ::DeviceNamePrint());

} // anonymous namespace
