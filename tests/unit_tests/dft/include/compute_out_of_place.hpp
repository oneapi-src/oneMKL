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

#ifndef ONEMKL_COMPUTE_OUT_OF_PLACE_HPP
#define ONEMKL_COMPUTE_OUT_OF_PLACE_HPP

#include "compute_tester.hpp"
#include <numeric>

template <oneapi::mkl::dft::precision precision, oneapi::mkl::dft::domain domain>
int DFT_Test<precision, domain>::test_out_of_place_buffer() {
    if (!init(MemoryAccessModel::buffer)) {
        return test_skipped;
    }

    auto [forward_distance, backward_distance] =
        get_default_distances<domain>(sizes, strides_fwd, strides_bwd);
    auto ref_distance = std::accumulate(sizes.begin(), sizes.end(), 1, std::multiplies<>());

    descriptor_t descriptor{ sizes };
    descriptor.set_value(oneapi::mkl::dft::config_param::PLACEMENT,
                         oneapi::mkl::dft::config_value::NOT_INPLACE);
    descriptor.set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, batches);
    descriptor.set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, forward_distance);
    descriptor.set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, backward_distance);
    if (strides_fwd.size()) {
        descriptor.set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, strides_fwd.data());
    }
    if (strides_bwd.size()) {
        descriptor.set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, strides_bwd.data());
    }
    else if constexpr (domain == oneapi::mkl::dft::domain::REAL) {
        const auto complex_strides = get_conjugate_even_complex_strides(sizes);
        descriptor.set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES,
                             complex_strides.data());
    }
    commit_descriptor(descriptor, sycl_queue);
    std::vector<FwdInputType> fwd_data(strided_copy(input, sizes, strides_fwd, batches));
    std::vector<FwdInputType> fwd_data_ref = fwd_data;

    auto tmp = std::vector<FwdOutputType>(
        cast_unsigned(backward_distance * batches + getdefault(strides_bwd, 0, 0L)), 0);
    {
        sycl::buffer<FwdInputType, 1> fwd_buf{ fwd_data };
        sycl::buffer<FwdOutputType, 1> bwd_buf{ tmp };

        oneapi::mkl::dft::compute_forward<descriptor_t, FwdInputType, FwdOutputType>(
            descriptor, fwd_buf, bwd_buf);

        {
            auto acc_bwd = bwd_buf.template get_host_access();
            auto bwd_ptr = acc_bwd.get_pointer();
            for (int64_t i = 0; i < batches; i++) {
                EXPECT_TRUE(check_equal_strided<domain == oneapi::mkl::dft::domain::REAL>(
                    bwd_ptr + backward_distance * i, out_host_ref.data() + ref_distance * i, sizes,
                    strides_bwd, abs_error_margin, rel_error_margin, std::cout));
            }
        }

        if (strides_bwd.size()) {
            descriptor.set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, strides_bwd.data());
        }
        else if constexpr (domain == oneapi::mkl::dft::domain::REAL) {
            const auto complex_strides = get_conjugate_even_complex_strides(sizes);
            descriptor.set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES,
                                 complex_strides.data());
        }
        else {
            auto real_strides = get_default_strides(sizes);
            descriptor.set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES,
                                 real_strides.data());
        }
        if (strides_fwd.size()) {
            descriptor.set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES,
                                 strides_fwd.data());
        }
        else {
            auto real_strides = get_default_strides(sizes);
            descriptor.set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES,
                                 real_strides.data());
        }
        commit_descriptor(descriptor, sycl_queue);

        oneapi::mkl::dft::compute_backward<std::remove_reference_t<decltype(descriptor)>,
                                           FwdOutputType, FwdInputType>(descriptor, bwd_buf,
                                                                        fwd_buf);
    }

    // account for scaling that occurs during DFT
    std::for_each(input.begin(), input.end(),
                  [this](auto &x) { x *= static_cast<PrecisionType>(forward_elements); });

    for (int64_t i = 0; i < batches; i++) {
        EXPECT_TRUE(check_equal_strided<false>(fwd_data.data() + forward_distance * i,
                                               input.data() + ref_distance * i, sizes, strides_fwd,
                                               abs_error_margin, rel_error_margin, std::cout));
    }

    return !::testing::Test::HasFailure();
}

template <oneapi::mkl::dft::precision precision, oneapi::mkl::dft::domain domain>
int DFT_Test<precision, domain>::test_out_of_place_USM() {
    if (!init(MemoryAccessModel::usm)) {
        return test_skipped;
    }
    const std::vector<sycl::event> no_dependencies;

    auto [forward_distance, backward_distance] =
        get_default_distances<domain>(sizes, strides_fwd, strides_bwd);

    descriptor_t descriptor{ sizes };
    descriptor.set_value(oneapi::mkl::dft::config_param::PLACEMENT,
                         oneapi::mkl::dft::config_value::NOT_INPLACE);
    descriptor.set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, batches);
    descriptor.set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, forward_distance);
    descriptor.set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, backward_distance);
    if (strides_fwd.size()) {
        descriptor.set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, strides_fwd.data());
    }
    if (strides_bwd.size()) {
        descriptor.set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, strides_bwd.data());
    }
    else if constexpr (domain == oneapi::mkl::dft::domain::REAL) {
        const auto complex_strides = get_conjugate_even_complex_strides(sizes);
        descriptor.set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES,
                             complex_strides.data());
    }
    commit_descriptor(descriptor, sycl_queue);

    auto ua_input = usm_allocator_t<FwdInputType>(cxt, *dev);
    auto ua_output = usm_allocator_t<FwdOutputType>(cxt, *dev);

    std::vector<FwdInputType, decltype(ua_input)> fwd(
        strided_copy(input, sizes, strides_fwd, batches, ua_input), ua_input);
    auto fwd_ref = fwd;
    std::vector<FwdOutputType, decltype(ua_output)> bwd(
        cast_unsigned(backward_distance * batches + getdefault(strides_bwd, 0, 0L)), ua_output);

    oneapi::mkl::dft::compute_forward<descriptor_t, FwdInputType, FwdOutputType>(
        descriptor, fwd.data(), bwd.data(), no_dependencies)
        .wait_and_throw();

    auto bwd_ptr = &bwd[0];
    auto ref_distance = std::accumulate(sizes.begin(), sizes.end(), 1, std::multiplies<>());
    for (int64_t i = 0; i < batches; i++) {
        EXPECT_TRUE(check_equal_strided<domain == oneapi::mkl::dft::domain::REAL>(
            bwd_ptr + backward_distance * i, out_host_ref.data() + ref_distance * i, sizes,
            strides_bwd, abs_error_margin, rel_error_margin, std::cout));
    }

    if (strides_bwd.size()) {
        descriptor.set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, strides_bwd.data());
    }
    else if constexpr (domain == oneapi::mkl::dft::domain::REAL) {
        const auto complex_strides = get_conjugate_even_complex_strides(sizes);
        descriptor.set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, complex_strides.data());
    }
    else {
        auto real_strides = get_default_strides(sizes);
        descriptor.set_value(oneapi::mkl::dft::config_param::INPUT_STRIDES, real_strides.data());
    }
    if (strides_fwd.size()) {
        descriptor.set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, strides_fwd.data());
    }
    else {
        auto real_strides = get_default_strides(sizes);
        descriptor.set_value(oneapi::mkl::dft::config_param::OUTPUT_STRIDES, real_strides.data());
    }
    commit_descriptor(descriptor, sycl_queue);

    oneapi::mkl::dft::compute_backward<std::remove_reference_t<decltype(descriptor)>, FwdOutputType,
                                       FwdInputType>(descriptor, bwd.data(), fwd.data(),
                                                     no_dependencies)
        .wait_and_throw();

    // account for scaling that occurs during DFT
    std::for_each(input.begin(), input.end(),
                  [this](auto &x) { x *= static_cast<PrecisionType>(forward_elements); });

    for (int64_t i = 0; i < batches; i++) {
        EXPECT_TRUE(check_equal_strided<false>(fwd.data() + forward_distance * i,
                                               input.data() + ref_distance * i, sizes, strides_fwd,
                                               abs_error_margin, rel_error_margin, std::cout));
    }

    return !::testing::Test::HasFailure();
}

#endif //ONEMKL_COMPUTE_OUT_OF_PLACE_HPP
