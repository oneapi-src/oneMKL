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

/**
 * @file cusolver_*.cpp : contain the implementation of all the routines
 * for CUDA backend
 */
#ifndef _CUSOLVER_HELPER_HPP_
#define _CUSOLVER_HELPER_HPP_
#include <CL/sycl.hpp>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cuda.h>
#include <complex>

#include "oneapi/mkl/types.hpp"
#include "runtime_support_helper.hpp"

namespace oneapi {
namespace mkl {
namespace lapack {
namespace cusolver {

// The static assert to make sure that all index types used in
// oneMKL/include/oneapi/mkl/lapack.hpp interface are int64_t
template <typename... Next>
struct is_int64 : std::false_type {};

template <typename First>
struct is_int64<First> : std::is_same<std::int64_t, First> {};

template <typename First, typename... Next>
struct is_int64<First, Next...>
        : std::integral_constant<bool, std::is_same<std::int64_t, First>::value &&
                                           is_int64<Next...>::value> {};

template <typename... T>
struct Overflow {
    static void inline check(T...) {}
};

template <typename Index, typename... T>
struct Overflow<Index, T...> {
    static void inline check(Index index, T... next) {
        if (std::abs(index) >= (1LL << 31)) {
            throw std::runtime_error(
                "Cusolver index overflow. Cusolver legacy API does not support 64 bit "
                "integer as data size. Thus, the data size should not be greater that "
                "maximum supported size by 32 bit integer.");
        }
        Overflow<T...>::check(next...);
    }
};

template <typename Index, typename... Next>
void overflow_check(Index index, Next... indices) {
    static_assert(is_int64<Index, Next...>::value, "oneMKL index type must be 64 bit integer.");
    Overflow<Index, Next...>::check(index, indices...);
}

class cusolver_error : virtual public std::runtime_error {
protected:
    inline const char *cusolver_error_map(cusolverStatus_t error) {
        switch (error) {
            case CUSOLVER_STATUS_SUCCESS: return "CUSOLVER_STATUS_SUCCESS";

            case CUSOLVER_STATUS_ALLOC_FAILED: return "CUSOLVER_STATUS_ALLOC_FAILED";

            case CUSOLVER_STATUS_INVALID_VALUE: return "CUSOLVER_STATUS_INVALID_VALUE";

            case CUSOLVER_STATUS_ARCH_MISMATCH: return "CUSOLVER_STATUS_ARCH_MISMATCH";

            case CUSOLVER_STATUS_EXECUTION_FAILED: return "CUSOLVER_STATUS_EXECUTION_FAILED";

            case CUSOLVER_STATUS_INTERNAL_ERROR: return "CUSOLVER_STATUS_INTERNAL_ERROR";

            case CUSOLVER_STATUS_NOT_INITIALIZED: return "CUSOLVER_STATUS_NOT_INITIALIZED";

            case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
                return "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";

            default: return "<unknown>";
        }
    }

    int error_number; ///< Error number
public:
    /** Constructor (C++ STL string, cusolverStatus_t).
   *  @param msg The error message
   *  @param err_num error number
   */
    explicit cusolver_error(std::string message, cusolverStatus_t result)
            : std::runtime_error((message + std::string(cusolver_error_map(result)))) {
        error_number = static_cast<int>(result);
    }

    /** Destructor.
   *  Virtual to allow for subclassing.
   */
    virtual ~cusolver_error() throw() {}

    /** Returns error number.
   *  @return #error_number
   */
    virtual int getErrorNumber() const throw() {
        return error_number;
    }
};

class cuda_error : virtual public std::runtime_error {
protected:
    inline const char *cuda_error_map(CUresult result) {
        switch (result) {
            case CUDA_SUCCESS: return "CUDA_SUCCESS";
            case CUDA_ERROR_NOT_PERMITTED: return "CUDA_ERROR_NOT_PERMITTED";
            case CUDA_ERROR_INVALID_CONTEXT: return "CUDA_ERROR_INVALID_CONTEXT";
            case CUDA_ERROR_INVALID_DEVICE: return "CUDA_ERROR_INVALID_DEVICE";
            case CUDA_ERROR_INVALID_VALUE: return "CUDA_ERROR_INVALID_VALUE";
            case CUDA_ERROR_OUT_OF_MEMORY: return "CUDA_ERROR_OUT_OF_MEMORY";
            case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES: return "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES";
            default: return "<unknown>";
        }
    }
    int error_number; ///< error number
public:
    /** Constructor (C++ STL string, CUresult).
   *  @param msg The error message
   *  @param err_num Error number
   */
    explicit cuda_error(std::string message, CUresult result)
            : std::runtime_error((message + std::string(cuda_error_map(result)))) {
        error_number = static_cast<int>(result);
    }

    /** Destructor.
   *  Virtual to allow for subclassing.
   */
    virtual ~cuda_error() throw() {}

    /** Returns error number.
   *  @return #error_number
   */
    virtual int getErrorNumber() const throw() {
        return error_number;
    }
};

#define CUDA_ERROR_FUNC(name, err, ...)                                 \
    err = name(__VA_ARGS__);                                            \
    if (err != CUDA_SUCCESS) {                                          \
        throw cuda_error(std::string(#name) + std::string(" : "), err); \
    }

#define CUSOLVER_ERROR_FUNC(name, err, ...)                                 \
    err = name(__VA_ARGS__);                                                \
    if (err != CUSOLVER_STATUS_SUCCESS) {                                   \
        throw cusolver_error(std::string(#name) + std::string(" : "), err); \
    }

inline cusolverEigType_t get_cusolver_itype(std::int64_t itype) {
    switch (itype) {
        case 1: return CUSOLVER_EIG_TYPE_1;
        case 2: return CUSOLVER_EIG_TYPE_2;
        case 3: return CUSOLVER_EIG_TYPE_3;
        default: throw "Wrong itype.";
    }
}

inline cusolverEigMode_t get_cusolver_job(oneapi::mkl::job jobz) {
    switch (jobz) {
        case oneapi::mkl::job::N: return CUSOLVER_EIG_MODE_NOVECTOR;
        case oneapi::mkl::job::V: return CUSOLVER_EIG_MODE_VECTOR;
        default: throw "Wrong jobz.";
    }
}

inline signed char get_cusolver_jobsvd(oneapi::mkl::jobsvd job) {
    switch (job) {
        case oneapi::mkl::jobsvd::N: return 'N';
        case oneapi::mkl::jobsvd::A: return 'A';
        case oneapi::mkl::jobsvd::O: return 'O';
        case oneapi::mkl::jobsvd::S: return 'S';
    }
}

inline cublasOperation_t get_cublas_operation(oneapi::mkl::transpose trn) {
    switch (trn) {
        case oneapi::mkl::transpose::nontrans: return CUBLAS_OP_N;
        case oneapi::mkl::transpose::trans: return CUBLAS_OP_T;
        case oneapi::mkl::transpose::conjtrans: return CUBLAS_OP_C;
        default: throw "Wrong transpose Operation.";
    }
}

inline cublasFillMode_t get_cublas_fill_mode(oneapi::mkl::uplo ul) {
    switch (ul) {
        case oneapi::mkl::uplo::upper: return CUBLAS_FILL_MODE_UPPER;
        case oneapi::mkl::uplo::lower: return CUBLAS_FILL_MODE_LOWER;
        default: throw "Wrong fill mode.";
    }
}

inline cublasSideMode_t get_cublas_side_mode(oneapi::mkl::side lr) {
    switch (lr) {
        case oneapi::mkl::side::left: return CUBLAS_SIDE_LEFT;
        case oneapi::mkl::side::right: return CUBLAS_SIDE_RIGHT;
        default: throw "Wrong side mode.";
    }
}

inline cublasSideMode_t get_cublas_generate(oneapi::mkl::generate qp) {
    switch (qp) {
        case oneapi::mkl::generate::Q: return CUBLAS_SIDE_LEFT;
        case oneapi::mkl::generate::P: return CUBLAS_SIDE_RIGHT;
        default: throw "Wrong generate.";
    }
}

/*converting std::complex<T> to cu<T>Complex*/
/*converting sycl::half to __half*/
template <typename T>
struct CudaEquivalentType {
    using Type = T;
};
template <>
struct CudaEquivalentType<sycl::half> {
    using Type = __half;
};
template <>
struct CudaEquivalentType<std::complex<float>> {
    using Type = cuComplex;
};
template <>
struct CudaEquivalentType<std::complex<double>> {
    using Type = cuDoubleComplex;
};

} // namespace cusolver
} // namespace lapack
} // namespace mkl
} // namespace oneapi
#endif // _CUSOLVER_HELPER_HPP_
