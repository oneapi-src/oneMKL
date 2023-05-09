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

// STL includes
#include <iostream>

// oneMKL/SYCL includes
#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif
#include "oneapi/mkl.hpp"

void run_example(const sycl::device& gpu_device) {
    constexpr std::size_t N = 10;

    // Catch asynchronous exceptions for gpu
    auto gpu_error_handler = [&](sycl::exception_list exceptions) {
        for (auto const& e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (sycl::exception const& e) {
                // Handle not dft related exceptions that happened during asynchronous call
                std::cerr << "Caught asynchronous SYCL exception:" << std::endl;
                std::cerr << "\t" << e.what() << std::endl;
            }
        }
        std::exit(2);
    };

    sycl::queue gpu_queue(gpu_device, gpu_error_handler);

    std::vector<std::complex<float>> input_data(N);
    std::vector<std::complex<float>> output_data(N);

    // enabling
    // 1. create descriptors
    oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE,
                                 oneapi::mkl::dft::domain::COMPLEX>
        desc(static_cast<std::int64_t>(N));

    // 2. variadic set_value
    desc.set_value(oneapi::mkl::dft::config_param::PLACEMENT,
                   oneapi::mkl::dft::config_value::NOT_INPLACE);
    desc.set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS,
                   static_cast<std::int64_t>(1));

    // 3. commit_descriptor (compile_time MKLGPU)
    desc.commit(oneapi::mkl::backend_selector<oneapi::mkl::backend::mklgpu>{ gpu_queue });

    // 4. compute_forward / compute_backward (MKLGPU)
    {
        sycl::buffer<std::complex<float>> input_buffer(input_data.data(), sycl::range<1>(N));
        sycl::buffer<std::complex<float>> output_buffer(output_data.data(), sycl::range<1>(N));
        oneapi::mkl::dft::compute_forward<decltype(desc), std::complex<float>, std::complex<float>>(
            desc, input_buffer, output_buffer);
    }
}

//
// Description of example setup, apis used and supported floating point type precisions
//
void print_example_banner() {
    std::cout << "\n"
                 "########################################################################\n"
                 "# Complex out-of-place forward transform for Buffer API's example:\n"
                 "#\n"
                 "# Using APIs:\n"
                 "#   Compile-time dispatch API\n"
                 "#   Buffer forward complex out-of-place\n"
                 "#\n"
                 "# Using single precision (float) data type\n"
                 "#\n"
                 "# For Intel GPU with Intel MKLGPU backend.\n"
                 "#\n"
                 "# The environment variable SYCL_DEVICE_FILTER can be used to specify\n"
                 "# SYCL device\n"
                 "########################################################################\n"
              << std::endl;
}

//
// Main entry point for example.
//
int main(int /*argc*/, char** /*argv*/) {
    print_example_banner();

    try {
        sycl::device gpu_device((sycl::gpu_selector_v));
        std::cout << "Running DFT Complex forward out-of-place buffer example" << std::endl;
        std::cout << "Using compile-time dispatch API with MKLGPU." << std::endl;
        std::cout << "Running with single precision real data type on:" << std::endl;
        std::cout << "\tGPU device :" << gpu_device.get_info<sycl::info::device::name>()
                  << std::endl;

        run_example(gpu_device);
        std::cout << "DFT Complex USM example ran OK on MKLGPU" << std::endl;
    }
    catch (sycl::exception const& e) {
        // Handle not dft related exceptions that happened during synchronous call
        std::cerr << "Caught synchronous SYCL exception:" << std::endl;
        std::cerr << "\t" << e.what() << std::endl;
        std::cerr << "\tSYCL error code: " << e.code().value() << std::endl;
        return 1;
    }
    catch (std::exception const& e) {
        // Handle not SYCL related exceptions that happened during synchronous call
        std::cerr << "Caught synchronous std::exception:" << std::endl;
        std::cerr << "\t" << e.what() << std::endl;
        return 1;
    }

    return 0;
}
