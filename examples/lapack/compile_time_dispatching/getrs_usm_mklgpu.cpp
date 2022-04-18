/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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
*       This example demonstrates use of oneapi::mkl::lapack::getrf and oneapi::mkl::lapack::getrs
*       to perform LU factorization and compute the solution on an Intel GPU SYCL device.
*
*       The supported floating point data types for matrix data are:
*           float
*
*******************************************************************************/

// STL includes
#include <iostream>
#include <complex>
#include <vector>

// oneMKL/SYCL includes
#include <CL/sycl.hpp>
#include "oneapi/mkl.hpp"

// #include "mkl.h"
// #include "oneapi/mkl/lapack.hpp"

// local includes
#include "example_helper.hpp"

//
// Main example for LU consisting of initialization of
// a general dense A matrix.
// Then the LU factorization
// A = P * L * U
// is performed followed by solving a system of linear
// equations using the computed LU factorization, with
// multiple right-hand sides.
// Finally the results are post processed.
//

void run_getrs_example(const sycl::device& device)
{
    // Matrix sizes and leading dimensions
    std::int64_t n    = 23;
    std::int64_t nhs  = 23;
    std::int64_t lda  = 32;
    std::int64_t ldb  = 32;

    // Variable holding status of calculations
    std::int64_t info = 0;

    // Asynchronous error handler
    auto error_handler = [&] (sycl::exception_list exceptions) {
        for (auto const& e : exceptions) {
            try {
                std::rethrow_exception(e);
            } catch(oneapi::mkl::lapack::exception const& e) {
                // Handle LAPACK related exceptions happened during asynchronous call
                info = e.info();
                std::cout << "Unexpected exception caught during asynchronous LAPACK operation:\n" << e.what() << "\ninfo: " << e.info() << std::endl;
            } catch(sycl::exception const& e) {
                // Handle not LAPACK related exceptions happened during asynchronous call
                std::cout << "Unexpected exception caught during asynchronous operation:\n" << e.what() << std::endl;
                info = -1;
            }
        }
    };

    // Create execution queue for selected device
    sycl::queue queue(device, error_handler);
    sycl::context context = queue.get_context();

    // Allocate matrices
    std::int64_t A_size = n * lda;
    std::int64_t B_size = nhs * ldb;
    std::int64_t ipiv_size = n;

    try {
        // USM
        float *A = (float *) sycl::malloc_shared(A_size * sizeof(float), device, context);
        float *B = (float *) sycl::malloc_shared(B_size * sizeof(float), device, context);
        std::int64_t *ipiv = (std::int64_t*) sycl::malloc_shared(ipiv_size * sizeof(std::int64_t), device, context);

        // Get sizes of scratchpads for calculations
        std::int64_t getrf_scratchpad_size = oneapi::mkl::lapack::getrf_scratchpad_size<float>(oneapi::mkl::backend_selector<oneapi::mkl::backend::mklgpu> {queue}, n, n, lda);
        std::int64_t getrs_scratchpad_size = oneapi::mkl::lapack::getrs_scratchpad_size<float>(oneapi::mkl::backend_selector<oneapi::mkl::backend::mklgpu> {queue}, oneapi::mkl::transpose::nontrans, n, nhs, lda, ldb);
        float *getrf_scratchpad = (float*) sycl::malloc_shared(getrf_scratchpad_size * sizeof(float), device, context);
        float *getrs_scratchpad = (float*) sycl::malloc_shared(getrs_scratchpad_size * sizeof(float), device, context);

        if (!A || !B || !ipiv || !getrf_scratchpad || !getrs_scratchpad)
            throw std::runtime_error("Failed to allocate USM memory.");

        // Initialize matrix A and B
        rand_matrix(A, oneapi::mkl::transpose::nontrans, n, n, lda);
        rand_matrix(B, oneapi::mkl::transpose::nontrans, n, nhs, ldb);

        // Perform factorization
        sycl::event getrf_done_event = oneapi::mkl::lapack::getrf(oneapi::mkl::backend_selector<oneapi::mkl::backend::mklgpu> {queue}, n, n, A, lda, ipiv, getrf_scratchpad, getrf_scratchpad_size);
        sycl::event getrs_done_event = oneapi::mkl::lapack::getrs(oneapi::mkl::backend_selector<oneapi::mkl::backend::mklgpu> {queue}, oneapi::mkl::transpose::nontrans, n, nhs, A, lda, ipiv, B, ldb, getrs_scratchpad, getrs_scratchpad_size, {getrf_done_event});

        // Wait until calculations are done
        queue.wait_and_throw();

        free(A, context);
        free(B, context);
        free(ipiv, context);
        free(getrf_scratchpad, context);
        free(getrs_scratchpad, context);

    } catch(oneapi::mkl::lapack::exception const& e) {
        // Handle LAPACK related exceptions happened during synchronous call
        std::cout << "Unexpected exception caught during synchronous call to LAPACK API:\nreason: " << e.what() << "\ninfo: " << e.info() << std::endl;
        info = e.info();
    } catch(sycl::exception const& e) {
        // Handle not LAPACK related exceptions happened during synchronous call
        std::cout << "Unexpected exception caught during synchronous call to SYCL API:\n" << e.what() << std::endl;
        info = -1;
    }

    std::cout << "getrs " << ((info == 0) ? "ran OK" : "FAILED") << std::endl;

    return;
}


//
// Description of example setup, apis used and supported floating point type precisions
//

void print_example_banner() {

    std::cout << "" << std::endl;
    std::cout << "########################################################################" << std::endl;
    std::cout << "# LU Factorization and Solve Example: " << std::endl;
    std::cout << "# " << std::endl;
    std::cout << "# Computes LU Factorization A = P * L * U" << std::endl;
    std::cout << "# and uses it to solve for X in a system of linear equations:" << std::endl;
    std::cout << "#   AX = B" << std::endl;
    std::cout << "# where A is a general dense matrix and B is a matrix whose columns" << std::endl; 
    std::cout << "# are the right-hand sides for the systems of equations." << std::endl;
    std::cout << "# " << std::endl;
    std::cout << "# Using apis:" << std::endl;
    std::cout << "#   getrf and getrs" << std::endl;
    std::cout << "# " << std::endl;
    std::cout << "# Supported floating point type precisions:" << std::endl;
    std::cout << "#   float" << std::endl;
    std::cout << "# " << std::endl;
    std::cout << "########################################################################" << std::endl;
    std::cout << std::endl;

}


//
// Main entry point for example.
//
int main(int argc, char **argv) {

    print_example_banner();

    sycl::device dev = sycl::device(sycl::gpu_selector());

    unsigned int vendor_id = static_cast<unsigned int>(dev.get_info<sycl::info::device::vendor_id>());
    if (dev.is_gpu() && vendor_id == INTEL_ID) {
        std::cout << "Running LAPACK getrs example on GPU device. Device name is: " << dev.get_info<sycl::info::device::name>() << ".\n";
    } else {
        std::cout << "FAILED: INTEL GPU device not found.\n";
        return 1;
    }

    std::cout << "\tRunning with single precision real data type:" << std::endl;
    run_getrs_example(dev);

    return 0;
}
