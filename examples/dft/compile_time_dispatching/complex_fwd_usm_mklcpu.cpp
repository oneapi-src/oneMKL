/*******************************************************************************
* Copyright 2022 Intel Corporation
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

/*
*
*  Content:
*       This example demonstrates use of oneapi::mkl::dft::getrf and
*       oneapi::mkl::dft::getrs to perform LU factorization and compute
*       the solution on both an Intel cpu device and NVIDIA cpu device.
*
*       This example demonstrates only single precision (float) data type
*       for matrix data
*
*******************************************************************************/

// STL includes
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <vector>

// oneMKL/SYCL includes
#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif
#include "oneapi/mkl.hpp"

// local includes
#include "example_helper.hpp"

void run_getrs_example(const sycl::device& cpu_device) {
    // Matrix sizes and leading dimensions
    constexpr std::size_t N = 10;
    std::int64_t rs[3] {0, N, 1};


    // Catch asynchronous exceptions for cpu and cpu
    auto cpu_error_handler = [&](sycl::exception_list exceptions) {
        for (auto const& e : exceptions) {
            try {
                std::rethrow_exception(e);
            }
            catch (sycl::exception const& e) {
                // Handle not dft related exceptions that happened during asynchronous call
                std::cerr
                    << "Caught asynchronous SYCL exception on cpu device during GETRF or GETRS:"
                    << std::endl;
                std::cerr << "\t" << e.what() << std::endl;
            }
        }
        std::exit(2);
    };

    std::cout << "DFTI example" << std::endl;
    //
    // Preparation on cpu
    //
    sycl::queue cpu_queue(cpu_device, cpu_error_handler);
    sycl::context cpu_context = cpu_queue.get_context();
    sycl::event cpu_getrf_done;

    double *x_usm = (double*) malloc_shared(N*2*sizeof(double), cpu_queue.get_device(), cpu_queue.get_context());

    // enabling
    // 1. create descriptors 
    oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::DOUBLE, oneapi::mkl::dft::domain::COMPLEX> desc(N);

    // 2. variadic set_value
    desc.set_value(oneapi::mkl::dft::config_param::PLACEMENT, oneapi::mkl::dft::config_value::NOT_INPLACE);

    // 3. commit_descriptor (compile_time CPU)
    desc.commit(oneapi::mkl::backend_selector<oneapi::mkl::backend::mklcpu>{ cpu_queue });

    // 5. compute_forward / compute_backward (CPU)
    // oneapi::mkl::dft::compute_forward(desc, x_usm);
}

//
// Description of example setup, apis used and supported floating point type precisions
//

void print_example_banner() {
    std::cout << "" << std::endl;
    std::cout << "########################################################################"
              << std::endl;
    std::cout
        << "# DFTI complex in-place forward transform for USM/Buffer API's example: "
        << std::endl;
    std::cout << "# " << std::endl;
    std::cout << "# Using APIs:" << std::endl;
    std::cout << "#   USM/BUffer forward complex in-place" << std::endl;
    std::cout << "# " << std::endl;
    std::cout << "# Using single precision (float) data type" << std::endl;
    std::cout << "# " << std::endl;
    std::cout << "# Device will be selected during runtime." << std::endl;
    std::cout << "# The environment variable SYCL_DEVICE_FILTER can be used to specify"
              << std::endl;
    std::cout << "# Using single precision (float) data type" << std::endl;
    std::cout << "# " << std::endl;
    std::cout << "# Running on both Intel cpu and NVIDIA cpu devices" << std::endl;
    std::cout << "# " << std::endl;
    std::cout << "########################################################################"
              << std::endl;
    std::cout << std::endl;
}

//
// Main entry point for example.
//
int main(int argc, char** argv) {
    print_example_banner();

    try {
        sycl::device cpu_dev((sycl::cpu_selector_v));
        std::cout << "Running DFT Complex forward inplace USM example" << std::endl;
        std::cout << "Running with single precision real data type on:" << std::endl;
        std::cout << "\tcpu device :" << cpu_dev.get_info<sycl::info::device::name>() << std::endl;

        run_getrs_example(cpu_dev);
        std::cout << "DFT Complex USM example ran OK on MKLcpu" << std::endl;
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
