/*******************************************************************************
* Copyright 2018-2022 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*******************************************************************************/

/*
*
*  Content:
*       This example demonstrates use of DPCPP API oneapi::mkl::blas::gemm
*       using unified shared memory to perform General
*       Matrix-Matrix Multiplication on an Intel CPU SYCL device.
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
        printf("GEMM_MKL_CPU\n");
        gemm_done = oneapi::mkl::blas::column_major::gemm(oneapi::mkl::backend_selector<oneapi::mkl::backend::mklcpu> {main_queue}, transA, transB, m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC);
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

    cl::sycl::device dev = cl::sycl::device(cl::sycl::cpu_selector());
    if (dev.is_cpu()) printf("Running on CPU device\n");

    std::cout << "Device name is: " << dev.get_info<cl::sycl::info::device::name>() << std::endl;

    std::cout << "\tRunning with single precision real data type:" << std::endl;
    run_gemm_example<float>(dev);

    return 0;
}
