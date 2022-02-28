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
*       This example demonstrates use of DPCPP API oneapi::mkl::blas::gemm
*       using unified shared memory to perform General
*       Matrix-Matrix Multiplication on a SYCL device (HOST, CPU, GPU) that
*       is selected during runtime.
*
*       C = alpha * op(A) * op(B) + beta * C
*
*       where op() is defined by one of oneapi::mkl::transpose::{nontrans,trans,conjtrans}
*
*
*       The supported floating point data types for gemm matrix data are:
*           float
*
*
*******************************************************************************/


// stl includes
#include <iostream>
#include <cstdlib>
#include <limits>
#include <vector>
#include <algorithm>
#include <cstring>
#include <list>
#include <iterator>


#include <CL/sycl.hpp>
#include "oneapi/mkl.hpp"
#include "oneapi/mkl/examples/common_helper_functions.hpp"

//
// Main example for Gemm consisting of
// initialization of A, B and C matrices as well as
// scalars alpha and beta.  Then the product
//
// C = alpha * op(A) * op(B) + beta * C
//
// is performed and finally the results are post processed.
//
template <typename fp>
void run_gemm_example(const cl::sycl::device &dev) {

    //
    // Initialize data for Gemm
    //
    // C = alpha * op(A) * op(B)  + beta * C
    //

    oneapi::mkl::transpose transA = oneapi::mkl::transpose::trans;
    oneapi::mkl::transpose transB = oneapi::mkl::transpose::nontrans;

    // matrix data sizes
    int m = 45;
    int n = 98;
    int k = 67;

    // leading dimensions of data
    int ldA = 103;
    int ldB = 105;
    int ldC = 106;
    int sizea, sizeb, sizec = ldC * n;

    // set scalar fp values
    fp alpha = set_fp_value(fp(2.0), fp(-0.5));
    fp beta  = set_fp_value(fp(3.0), fp(-1.5));

    // Catch asynchronous exceptions
    auto exception_handler = [] (cl::sycl::exception_list exceptions) {
        for (std::exception_ptr const& e : exceptions) {
            try {
                std::rethrow_exception(e);
            } catch(cl::sycl::exception const& e) {
                std::cout << "Caught asynchronous SYCL exception during GEMM:\n"
                << e.what() << std::endl;
            }
        }
    };

    // create execution queue and buffers of matrix data
    cl::sycl::queue main_queue(dev, exception_handler);
    cl::sycl::event gemm_done;
    std::vector<cl::sycl::event> gemm_dependencies;
    cl::sycl::context cxt = main_queue.get_context();
    sizea = (transA == oneapi::mkl::transpose::nontrans) ? ldA * k : ldA * m;
    sizeb = (transB == oneapi::mkl::transpose::nontrans) ? ldB * n : ldB * k;

    auto A = (fp *) malloc_shared(sizea * sizeof(fp), dev, cxt);
    auto B = (fp *) malloc_shared(sizeb * sizeof(fp), dev, cxt);
    auto C = (fp *) malloc_shared(sizec * sizeof(fp), dev, cxt);

    if (!A || !B || !C)
        throw std::runtime_error("Failed to allocate USM memory.");

    rand_matrix(A, transA, m, k, ldA);
    rand_matrix(B, transB, k, n, ldB);
    rand_matrix(C, oneapi::mkl::transpose::nontrans, m, n, ldC);


    //
    // Execute Gemm
    //

    // add oneapi::mkl::blas::gemm to execution queue
    try {
        printf("Runtime compilation, backend not specified\n");
        gemm_done = oneapi::mkl::blas::column_major::gemm(main_queue, transA, transB, m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC);
    }
    catch(cl::sycl::exception const& e) {
        std::cout << "\t\tCaught synchronous SYCL exception during GEMM:\n"
                  << e.what() << std::endl << "OpenCL status: " << get_error_code(e) << std::endl;
    }

    gemm_done.wait();

    //
    // Post Processing
    //

    std::cout << "\n\t\tGEMM parameters:\n";
    std::cout << "\t\t\ttransA = " << ( transA == oneapi::mkl::transpose::nontrans ? "nontrans" : ( transA == oneapi::mkl::transpose::trans ? "trans" : "conjtrans"))
              <<   ", transB = " << ( transB == oneapi::mkl::transpose::nontrans ? "nontrans" : ( transB == oneapi::mkl::transpose::trans ? "trans" : "conjtrans")) << std::endl;
    std::cout << "\t\t\tm = " << m << ", n = " << n << ", k = " << k << std::endl;
    std::cout << "\t\t\tlda = " << ldA << ", ldB = " << ldB << ", ldC = " << ldC << std::endl;
    std::cout << "\t\t\talpha = " << alpha << ", beta = " << beta << std::endl;


    std::cout << "\n\t\tOutputting 2x2 block of A,B,C matrices:" << std::endl;

    // output the top 2x2 block of A matrix
    print_2x2_matrix_values(A, ldA, "A");

    // output the top 2x2 block of B matrix
    print_2x2_matrix_values(B, ldB, "B");

    // output the top 2x2 block of C matrix
    print_2x2_matrix_values(C, ldC, "C");

    free(A, cxt);
    free(B, cxt);
    free(C, cxt);
}

//
// Description of example setup, apis used and supported floating point type precisions
//
void print_example_banner() {

    std::cout << "" << std::endl;
    std::cout << "########################################################################" << std::endl;
    std::cout << "# General Matrix-Matrix Multiplication using Unified Shared Memory Example: " << std::endl;
    std::cout << "# " << std::endl;
    std::cout << "# C = alpha * A * B + beta * C" << std::endl;
    std::cout << "# " << std::endl;
    std::cout << "# where A, B and C are general dense matrices and alpha, beta are" << std::endl;
    std::cout << "# floating point type precision scalars." << std::endl;
    std::cout << "# " << std::endl;
    std::cout << "# Using apis:" << std::endl;
    std::cout << "#   gemm" << std::endl;
    std::cout << "# " << std::endl;
    std::cout << "# Supported floating point type precisions:" << std::endl;
    std::cout << "#   float" << std::endl;
    std::cout << "# " << std::endl;
    std::cout << "########################################################################" << std::endl;
    std::cout << std::endl;

}

//
// Main entry point for example
//
int main (int argc, char ** argv) {
    print_example_banner();

    cl::sycl::device dev = cl::sycl::device(cl::sycl::default_selector());


    if (dev.is_gpu()) {
        printf("Running on GPU device\n");
        unsigned int vendor_id = static_cast<unsigned int>(dev.get_info<cl::sycl::info::device::vendor_id>());
        if (vendor_id == INTEL_ID) {
            bool is_level0 = dev.get_info<cl::sycl::info::device::opencl_c_version>().empty();
            if (is_level0) printf("DPC++ running with Level0 backend\n");
            else printf("DPC++ running with OpenCL backend\n");
        }
    } else {
        printf("Running on CPU device\n");
    }

    std::cout << "Device name is: " << dev.get_info<cl::sycl::info::device::name>() << ".\n";

    std::cout << "\tRunning with single precision real data type:" << std::endl;
    run_gemm_example<float>(dev);

    return 0;
}
