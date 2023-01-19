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

/* Note: There is no implementation for Domain Real */
template <oneapi::mkl::dft::precision precision, oneapi::mkl::dft::domain domain>
int DFT_Test<precision, domain>::test_out_of_place_buffer() {
    if (!init(TestType::buffer)) {
        return test_skipped;
    }

    descriptor_t descriptor{ size };
    descriptor.set_value(oneapi::mkl::dft::config_param::PLACEMENT,
                         oneapi::mkl::dft::config_value::NOT_INPLACE);
    commit_descriptor(descriptor, sycl_queue);

    std::vector<OutputType> out_host(size);
    std::vector<InputType> out_host_back(size);
    sycl::buffer<InputType, 1> in_dev{ sycl::range<1>(size) };
    sycl::buffer<OutputType, 1> out_dev{ sycl::range<1>(size) };
    sycl::buffer<InputType, 1> out_back_dev{ sycl::range<1>(size) };

    copy_to_device(sycl_queue, input, in_dev);

    try {
        oneapi::mkl::dft::compute_forward<descriptor_t, InputType, OutputType>(descriptor, in_dev,
                                                                               out_dev);
    }
    catch (oneapi::mkl::unimplemented &e) {
        std::cout << "Skipping test because: \"" << e.what() << "\"" << std::endl;
        return test_skipped;
    }

    copy_to_host(sycl_queue, out_dev, out_host);

    EXPECT_TRUE(check_equal_vector(out_host.data(), out_host_ref.data(), out_host.size(),
                                   error_margin, std::cout));

    descriptor_t descriptor_back{ size };
    descriptor_back.set_value(oneapi::mkl::dft::config_param::PLACEMENT,
                              oneapi::mkl::dft::config_value::NOT_INPLACE);
    descriptor_back.set_value(oneapi::mkl::dft::config_param::BACKWARD_SCALE, (1.0 / size));
    commit_descriptor(descriptor_back, sycl_queue);

    try {
        oneapi::mkl::dft::compute_backward<std::remove_reference_t<decltype(descriptor_back)>,
                                           OutputType, InputType>(descriptor_back, out_dev,
                                                                  out_back_dev);
    }
    catch (oneapi::mkl::unimplemented &e) {
        std::cout << "Skipping test because: \"" << e.what() << "\"" << std::endl;
        return test_skipped;
    }

    copy_to_host(sycl_queue, out_back_dev, out_host_back);

    EXPECT_TRUE(check_equal_vector(out_host_back.data(), input.data(), input.size(), error_margin,
                                   std::cout));

    return !::testing::Test::HasFailure();
}

template <oneapi::mkl::dft::precision precision, oneapi::mkl::dft::domain domain>
int DFT_Test<precision, domain>::test_out_of_place_USM() {
    if (!init(TestType::usm)) {
        return test_skipped;
    }

    descriptor_t descriptor{ size };
    descriptor.set_value(oneapi::mkl::dft::config_param::PLACEMENT,
                         oneapi::mkl::dft::config_value::NOT_INPLACE);
    commit_descriptor(descriptor, sycl_queue);

    std::vector<OutputType> out_host(size);
    std::vector<InputType> out_host_back(size);

    auto ua_input = usm_allocator_t<InputType>(cxt, *dev);
    auto ua_output = usm_allocator_t<OutputType>(cxt, *dev);

    std::vector<InputType, decltype(ua_input)> in(size, ua_input);
    std::vector<OutputType, decltype(ua_output)> out(size, ua_output);

    std::copy(input.begin(), input.end(), in.begin());

    try {
        std::vector<sycl::event> dependencies;
        sycl::event done = oneapi::mkl::dft::compute_forward<descriptor_t, InputType, OutputType>(
            descriptor, in.data(), out.data(), dependencies);
        done.wait();
    }
    catch (oneapi::mkl::unimplemented &e) {
        std::cout << "Skipping test because: \"" << e.what() << "\"" << std::endl;
        return test_skipped;
    }

    EXPECT_TRUE(
        check_equal_vector(out.data(), out_host_ref.data(), out.size(), error_margin, std::cout));

    descriptor_t descriptor_back{ size };
    descriptor_back.set_value(oneapi::mkl::dft::config_param::PLACEMENT,
                              oneapi::mkl::dft::config_value::NOT_INPLACE);
    descriptor_back.set_value(oneapi::mkl::dft::config_param::BACKWARD_SCALE, (1.0 / size));
    commit_descriptor(descriptor_back, sycl_queue);

    std::vector<InputType, decltype(ua_input)> out_back(size, ua_input);

    try {
        std::vector<sycl::event> dependencies;
        sycl::event done =
            oneapi::mkl::dft::compute_backward<std::remove_reference_t<decltype(descriptor_back)>,
                                               OutputType, InputType>(descriptor_back, out.data(),
                                                                      out_back.data());
        done.wait();
    }
    catch (oneapi::mkl::unimplemented &e) {
        std::cout << "Skipping test because: \"" << e.what() << "\"" << std::endl;
        return test_skipped;
    }

    EXPECT_TRUE(
        check_equal_vector(out_back.data(), input.data(), input.size(), error_margin, std::cout));

    return !::testing::Test::HasFailure();
}

#endif //ONEMKL_COMPUTE_OUT_OF_PLACE_HPP
