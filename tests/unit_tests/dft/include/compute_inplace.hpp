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

#ifndef ONEMKL_COMPUTE_INPLACE_HPP
#define ONEMKL_COMPUTE_INPLACE_HPP

#include "compute_tester.hpp"

template <oneapi::mkl::dft::precision precision, oneapi::mkl::dft::domain domain>
int DFT_Test<precision, domain>::test_in_place_buffer() {
    if (!init(MemoryAccessModel::buffer)) {
        return test_skipped;
    }

    descriptor_t descriptor{ size };
    descriptor.set_value(oneapi::mkl::dft::config_param::PLACEMENT,
                         oneapi::mkl::dft::config_value::INPLACE);

    const size_t container_size =
        domain == oneapi::mkl::dft::domain::REAL ? conjugate_even_size : size;

    std::vector<FwdInputType> inout_host(container_size, static_cast<FwdInputType>(0));
    std::copy(input.cbegin(), input.cend(), inout_host.begin());
    sycl::buffer<FwdInputType, 1> inout_buf{ inout_host.data(), sycl::range<1>(container_size) };

    commit_descriptor(descriptor, sycl_queue);

    try {
        oneapi::mkl::dft::compute_forward<descriptor_t, FwdInputType>(descriptor, inout_buf);
    }
    catch (oneapi::mkl::unimplemented &e) {
        std::cout << "Skipping test because: \"" << e.what() << "\"" << std::endl;
        return test_skipped;
    }

    if constexpr (domain == oneapi::mkl::dft::domain::REAL) {
        std::vector<FwdInputType> out_host_ref_conjugate =
            std::vector<FwdInputType>(conjugate_even_size);
        for (int i = 0; i < out_host_ref_conjugate.size(); i += 2) {
            out_host_ref_conjugate[i] = out_host_ref[i / 2].real();
            out_host_ref_conjugate[i + 1] = out_host_ref[i / 2].imag();
        }
        auto acc_host = inout_buf.template get_host_access();
        EXPECT_TRUE(check_equal_vector(acc_host.get_pointer(), out_host_ref_conjugate.data(),
                                       inout_host.size(), abs_error_margin, rel_error_margin, std::cout));
    }
    else {
        auto acc_host = inout_buf.template get_host_access();
        EXPECT_TRUE(check_equal_vector(acc_host.get_pointer(), out_host_ref.data(),
                                       inout_host.size(), abs_error_margin, rel_error_margin, std::cout));
    }

    descriptor_t descriptor_back{ size };
    descriptor_back.set_value(oneapi::mkl::dft::config_param::PLACEMENT,
                              oneapi::mkl::dft::config_value::INPLACE);
    descriptor_back.set_value(oneapi::mkl::dft::config_param::BACKWARD_SCALE, (1.0 / size));
    commit_descriptor(descriptor_back, sycl_queue);

    try {
        oneapi::mkl::dft::compute_backward<std::remove_reference_t<decltype(descriptor_back)>,
                                           FwdInputType>(descriptor_back, inout_buf);
    }
    catch (oneapi::mkl::unimplemented &e) {
        std::cout << "Skipping test because: \"" << e.what() << "\"" << std::endl;
        return test_skipped;
    }

    {
        auto acc_host = inout_buf.template get_host_access();
        EXPECT_TRUE(check_equal_vector(acc_host.get_pointer(), input.data(), input.size(),
                                       abs_error_margin, rel_error_margin, std::cout));
    }
    return !::testing::Test::HasFailure();
}

template <oneapi::mkl::dft::precision precision, oneapi::mkl::dft::domain domain>
int DFT_Test<precision, domain>::test_in_place_USM() {
    if (!init(MemoryAccessModel::usm)) {
        return test_skipped;
    }

    descriptor_t descriptor{ size };
    descriptor.set_value(oneapi::mkl::dft::config_param::PLACEMENT,
                         oneapi::mkl::dft::config_value::INPLACE);
    commit_descriptor(descriptor, sycl_queue);

    const size_t container_size =
        domain == oneapi::mkl::dft::domain::REAL ? conjugate_even_size : size;

    auto ua_input = usm_allocator_t<FwdInputType>(cxt, *dev);

    std::vector<FwdInputType, decltype(ua_input)> inout(container_size, ua_input);
    std::copy(input.begin(), input.end(), inout.begin());

    try {
        std::vector<sycl::event> dependencies;
        sycl::event done = oneapi::mkl::dft::compute_forward<descriptor_t, FwdInputType>(
            descriptor, inout.data(), dependencies);
        done.wait();
    }
    catch (oneapi::mkl::unimplemented &e) {
        std::cout << "Skipping test because: \"" << e.what() << "\"" << std::endl;
        return test_skipped;
    }

    if constexpr (domain == oneapi::mkl::dft::domain::REAL) {
        std::vector<FwdInputType> out_host_ref_conjugate =
            std::vector<FwdInputType>(conjugate_even_size);
        for (int i = 0; i < out_host_ref_conjugate.size(); i += 2) {
            out_host_ref_conjugate[i] = out_host_ref[i / 2].real();
            out_host_ref_conjugate[i + 1] = out_host_ref[i / 2].imag();
        }
        EXPECT_TRUE(check_equal_vector(inout.data(), out_host_ref_conjugate.data(), inout.size(),
                                       abs_error_margin, rel_error_margin, std::cout));
    }
    else {
        EXPECT_TRUE(check_equal_vector(inout.data(), out_host_ref.data(), inout.size(),
                                       abs_error_margin, rel_error_margin, std::cout));
    }

    descriptor_t descriptor_back{ size };
    descriptor_back.set_value(oneapi::mkl::dft::config_param::PLACEMENT,
                              oneapi::mkl::dft::config_value::INPLACE);
    descriptor_back.set_value(oneapi::mkl::dft::config_param::BACKWARD_SCALE, (1.0 / size));
    commit_descriptor(descriptor_back, sycl_queue);

    try {
        std::vector<sycl::event> dependencies;
        sycl::event done =
            oneapi::mkl::dft::compute_backward<std::remove_reference_t<decltype(descriptor_back)>,
                                               FwdInputType>(descriptor_back, inout.data());
        done.wait();
    }
    catch (oneapi::mkl::unimplemented &e) {
        std::cout << "Skipping test because: \"" << e.what() << "\"" << std::endl;
        return test_skipped;
    }

    EXPECT_TRUE(
        check_equal_vector(inout.data(), input.data(), input.size(), abs_error_margin, rel_error_margin, std::cout));

    return !::testing::Test::HasFailure();
}

#endif //ONEMKL_COMPUTE_INPLACE_HPP
