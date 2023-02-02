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
    if (!init(MemoryAccessModel::buffer)) {
        return test_skipped;
    }

    const size_t bwd_size = domain == oneapi::mkl::dft::domain::REAL ? (size / 2) + 1 : size;

    descriptor_t descriptor{ size };
    descriptor.set_value(oneapi::mkl::dft::config_param::PLACEMENT,
                         oneapi::mkl::dft::config_value::NOT_INPLACE);
    commit_descriptor(descriptor, sycl_queue);

    descriptor_t descriptor_back{ size };
    descriptor_back.set_value(oneapi::mkl::dft::config_param::PLACEMENT,
                              oneapi::mkl::dft::config_value::NOT_INPLACE);
    descriptor_back.set_value(oneapi::mkl::dft::config_param::BACKWARD_SCALE, (1.0 / size));
    commit_descriptor(descriptor_back, sycl_queue);

    std::vector<FwdInputType> fwd_data(input);
    std::vector<FwdOutputType> bwd_data(bwd_size, 0);

    {
        sycl::buffer<FwdInputType, 1> fwd_buf{ fwd_data };
        sycl::buffer<FwdOutputType, 1> bwd_buf{ bwd_data };

        try {
            oneapi::mkl::dft::compute_forward<descriptor_t, FwdInputType, FwdOutputType>(
                descriptor, fwd_buf, bwd_buf);
        }
        catch (oneapi::mkl::unimplemented &e) {
            std::cout << "Skipping test because: \"" << e.what() << "\"" << std::endl;
            return test_skipped;
        }

        {
            auto acc_bwd = bwd_buf.template get_host_access();
            EXPECT_TRUE(check_equal_vector(acc_bwd.get_pointer(), out_host_ref.data(),
                                           bwd_data.size(), abs_error_margin, rel_error_margin,
                                           std::cout));
        }

        try {
            oneapi::mkl::dft::compute_backward<std::remove_reference_t<decltype(descriptor_back)>,
                                               FwdOutputType, FwdInputType>(descriptor_back,
                                                                            bwd_buf, fwd_buf);
        }
        catch (oneapi::mkl::unimplemented &e) {
            std::cout << "Skipping test because: \"" << e.what() << "\"" << std::endl;
            return test_skipped;
        }
    }

    EXPECT_TRUE(check_equal_vector(fwd_data.data(), input.data(), input.size(), abs_error_margin,
                                   rel_error_margin, std::cout));
    return !::testing::Test::HasFailure();
}

template <oneapi::mkl::dft::precision precision, oneapi::mkl::dft::domain domain>
int DFT_Test<precision, domain>::test_out_of_place_USM() {
    if (!init(MemoryAccessModel::usm)) {
        return test_skipped;
    }
    const std::vector<sycl::event> no_dependencies;

    const size_t bwd_size = domain == oneapi::mkl::dft::domain::REAL ? (size / 2) + 1 : size;

    descriptor_t descriptor{ size };
    descriptor.set_value(oneapi::mkl::dft::config_param::PLACEMENT,
                         oneapi::mkl::dft::config_value::NOT_INPLACE);
    commit_descriptor(descriptor, sycl_queue);

    descriptor_t descriptor_back{ size };
    descriptor_back.set_value(oneapi::mkl::dft::config_param::PLACEMENT,
                              oneapi::mkl::dft::config_value::NOT_INPLACE);
    descriptor_back.set_value(oneapi::mkl::dft::config_param::BACKWARD_SCALE, (1.0 / size));
    commit_descriptor(descriptor_back, sycl_queue);

    auto ua_input = usm_allocator_t<FwdInputType>(cxt, *dev);
    auto ua_output = usm_allocator_t<FwdOutputType>(cxt, *dev);

    std::vector<FwdInputType, decltype(ua_input)> fwd(input.begin(), input.end(), ua_input);
    std::vector<FwdOutputType, decltype(ua_output)> bwd(bwd_size, ua_output);

    try {
        oneapi::mkl::dft::compute_forward<descriptor_t, FwdInputType, FwdOutputType>(
            descriptor, fwd.data(), bwd.data(), no_dependencies)
            .wait();
    }
    catch (oneapi::mkl::unimplemented &e) {
        std::cout << "Skipping test because: \"" << e.what() << "\"" << std::endl;
        return test_skipped;
    }

    EXPECT_TRUE(check_equal_vector(bwd.data(), out_host_ref.data(), bwd.size(), abs_error_margin,
                                   rel_error_margin, std::cout));

    try {
        oneapi::mkl::dft::compute_backward<std::remove_reference_t<decltype(descriptor_back)>,
                                           FwdOutputType, FwdInputType>(descriptor_back, bwd.data(),
                                                                        fwd.data(), no_dependencies)
            .wait();
    }
    catch (oneapi::mkl::unimplemented &e) {
        std::cout << "Skipping test because: \"" << e.what() << "\"" << std::endl;
        return test_skipped;
    }

    EXPECT_TRUE(check_equal_vector(fwd.data(), input.data(), input.size(), abs_error_margin,
                                   rel_error_margin, std::cout));

    return !::testing::Test::HasFailure();
}

#endif //ONEMKL_COMPUTE_OUT_OF_PLACE_HPP
