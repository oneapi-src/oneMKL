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

#ifndef ONEMKL_COMPUTE_OUT_OF_PLACE_REAL_REAL_HPP
#define ONEMKL_COMPUTE_OUT_OF_PLACE_REAL_REAL_HPP

#include "compute_tester.hpp"

/* Test is not implemented because currently there are no available dft implementations.
 * These are stubs to make sure that dft::oneapi::mkl::unimplemented exception is thrown */
template <oneapi::mkl::dft::precision precision, oneapi::mkl::dft::domain domain>
int DFT_Test<precision, domain>::test_out_of_place_real_real_USM() {
    if (!init(MemoryAccessModel::usm)) {
        return test_skipped;
    }

    try {
        descriptor_t descriptor{ size };

        descriptor.set_value(oneapi::mkl::dft::config_param::PLACEMENT,
                             oneapi::mkl::dft::config_value::NOT_INPLACE);
        descriptor.set_value(oneapi::mkl::dft::config_param::COMPLEX_STORAGE,
                             oneapi::mkl::dft::config_value::REAL_REAL);
        commit_descriptor(descriptor, sycl_queue);

        auto ua_input = usm_allocator_t<PrecisionType>(cxt, *dev);
        auto ua_output = usm_allocator_t<PrecisionType>(cxt, *dev);

        std::vector<PrecisionType, decltype(ua_input)> in_re(size, ua_input);
        std::vector<PrecisionType, decltype(ua_input)> in_im(size, ua_input);
        std::vector<PrecisionType, decltype(ua_output)> out_re(size, ua_output);
        std::vector<PrecisionType, decltype(ua_output)> out_im(size, ua_output);
        std::vector<PrecisionType, decltype(ua_input)> out_back_re(size, ua_input);
        std::vector<PrecisionType, decltype(ua_input)> out_back_im(size, ua_input);

        std::copy(input_re.begin(), input_re.end(), in_re.begin());
        std::copy(input_im.begin(), input_im.end(), in_im.begin());

        std::vector<sycl::event> dependencies;
        sycl::event done =
            oneapi::mkl::dft::compute_forward<descriptor_t, PrecisionType, PrecisionType>(
                descriptor, in_re.data(), in_im.data(), out_re.data(), out_im.data(), dependencies);
        done.wait();

        descriptor_t descriptor_back{ size };

        descriptor_back.set_value(oneapi::mkl::dft::config_param::PLACEMENT,
                                  oneapi::mkl::dft::config_value::NOT_INPLACE);
        descriptor_back.set_value(oneapi::mkl::dft::config_param::COMPLEX_STORAGE,
                                  oneapi::mkl::dft::config_value::REAL_REAL);
        descriptor_back.set_value(oneapi::mkl::dft::config_param::BACKWARD_SCALE, (1.0 / size));
        commit_descriptor(descriptor_back, sycl_queue);

        done =
            oneapi::mkl::dft::compute_backward<std::remove_reference_t<decltype(descriptor_back)>,
                                               PrecisionType, PrecisionType>(
                descriptor_back, out_re.data(), out_im.data(), out_back_re.data(),
                out_back_im.data());
        done.wait();
    }
    catch (oneapi::mkl::unimplemented &e) {
        std::cout << "Skipping test because: \"" << e.what() << "\"" << std::endl;
        return test_skipped;
    }

    /* Once implementations exist, results will need to be verified */
    EXPECT_TRUE(false);

    return !::testing::Test::HasFailure();
}

/* Test is not implemented because currently there are no available dft implementations.
 * These are stubs to make sure that dft::oneapi::mkl::unimplemented exception is thrown */
template <oneapi::mkl::dft::precision precision, oneapi::mkl::dft::domain domain>
int DFT_Test<precision, domain>::test_out_of_place_real_real_buffer() {
    if (!init(MemoryAccessModel::buffer)) {
        return test_skipped;
    }

    try {
        descriptor_t descriptor{ size };

        descriptor.set_value(oneapi::mkl::dft::config_param::PLACEMENT,
                             oneapi::mkl::dft::config_value::NOT_INPLACE);
        descriptor.set_value(oneapi::mkl::dft::config_param::COMPLEX_STORAGE,
                             oneapi::mkl::dft::config_value::REAL_REAL);
        commit_descriptor(descriptor, sycl_queue);

        sycl::buffer<PrecisionType, 1> in_dev_re{ input_re.data(), sycl::range<1>(size) };
        sycl::buffer<PrecisionType, 1> in_dev_im{ input_im.data(), sycl::range<1>(size) };
        sycl::buffer<PrecisionType, 1> out_dev_re{ sycl::range<1>(size) };
        sycl::buffer<PrecisionType, 1> out_dev_im{ sycl::range<1>(size) };
        sycl::buffer<PrecisionType, 1> out_back_dev_re{ sycl::range<1>(size) };
        sycl::buffer<PrecisionType, 1> out_back_dev_im{ sycl::range<1>(size) };

        oneapi::mkl::dft::compute_forward<descriptor_t, PrecisionType, PrecisionType>(
            descriptor, in_dev_re, in_dev_im, out_dev_re, out_dev_im);

        descriptor_t descriptor_back{ size };

        descriptor_back.set_value(oneapi::mkl::dft::config_param::PLACEMENT,
                                  oneapi::mkl::dft::config_value::NOT_INPLACE);
        descriptor_back.set_value(oneapi::mkl::dft::config_param::COMPLEX_STORAGE,
                                  oneapi::mkl::dft::config_value::REAL_REAL);
        descriptor_back.set_value(oneapi::mkl::dft::config_param::BACKWARD_SCALE, (1.0 / size));
        commit_descriptor(descriptor_back, sycl_queue);

        oneapi::mkl::dft::compute_backward<std::remove_reference_t<decltype(descriptor_back)>,
                                           PrecisionType, PrecisionType>(
            descriptor_back, out_dev_re, out_dev_im, out_back_dev_re, out_back_dev_im);
    }
    catch (oneapi::mkl::unimplemented &e) {
        std::cout << "Skipping test because: \"" << e.what() << "\"" << std::endl;
        return test_skipped;
    }

    /* Once implementations exist, results will need to be verified */
    EXPECT_TRUE(false);

    return !::testing::Test::HasFailure();
}

#endif //ONEMKL_COMPUTE_OUT_OF_PLACE_REAL_REAL_HPP
