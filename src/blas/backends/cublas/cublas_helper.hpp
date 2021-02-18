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
 * @file mkl_blas_cublas.cpp : contains the implementation of all the routines
 * for CUDA backend
 */
#ifndef _MKL_BLAS_CUBLAS_HELPER_HPP_
#define _MKL_BLAS_CUBLAS_HELPER_HPP_
#include <cublas_v2.h>
#include <cuda.h>
#include <complex>
#include "oneapi/mkl/types.hpp"

namespace oneapi {
namespace mkl {
namespace blas {
namespace cublas {

// The static assert to make sure that all index types used in
// src/oneMKL/backend/cublas/blas.hpp interface are int64_t
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
                "Cublas index overflow. cublas does not support 64 bit integer as "
                "data size. Thus, the data size should not be greater that maximum "
                "supported size by 32 bit integer.");
        }
        Overflow<T...>::check(next...);
    }
};

template <typename Index, typename... Next>
void overflow_check(Index index, Next... indices) {
    static_assert(is_int64<Index, Next...>::value, "oneMKL index type must be 64 bit integer.");
    Overflow<Index, Next...>::check(index, indices...);
}

class cublas_error : virtual public std::runtime_error {
protected:
    inline const char *cublas_error_map(cublasStatus_t error) {
        switch (error) {
            case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";

            case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";

            case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";

            case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";

            case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";

            case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";

            case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";

            case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";

            case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";

            case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";

            default: return "<unknown>";
        }
    }

    int error_number; ///< Error number
public:
    /** Constructor (C++ STL string, cublasStatus_t).
   *  @param msg The error message
   *  @param err_num error number
   */
    explicit cublas_error(std::string message, cublasStatus_t result)
            : std::runtime_error((message + std::string(cublas_error_map(result)))) {
        error_number = static_cast<int>(result);
    }

    /** Destructor.
   *  Virtual to allow for subclassing.
   */
    virtual ~cublas_error() throw() {}

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

#define CUBLAS_ERROR_FUNC(name, err, ...)                                 \
    err = name(__VA_ARGS__);                                              \
    if (err != CUBLAS_STATUS_SUCCESS) {                                   \
        throw cublas_error(std::string(#name) + std::string(" : "), err); \
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

inline cublasDiagType_t get_cublas_diag_type(oneapi::mkl::diag un) {
    switch (un) {
        case oneapi::mkl::diag::unit: return CUBLAS_DIAG_UNIT;
        case oneapi::mkl::diag::nonunit: return CUBLAS_DIAG_NON_UNIT;
        default: throw "Wrong diag type.";
    }
}

inline cublasSideMode_t get_cublas_side_mode(oneapi::mkl::side lr) {
    switch (lr) {
        case oneapi::mkl::side::left: return CUBLAS_SIDE_LEFT;
        case oneapi::mkl::side::right: return CUBLAS_SIDE_RIGHT;
        default: throw "Wrong side mode.";
    }
}

/*converting std::complex<T> to cu<T>Complex*/
template <typename T>
struct CudaEquivalentType {
    using Type = T;
};

template <>
struct CudaEquivalentType<std::complex<float>> {
    using Type = cuComplex;
};
template <>
struct CudaEquivalentType<std::complex<double>> {
    using Type = cuDoubleComplex;
};

} // namespace cublas
} // namespace blas
} // namespace mkl
} // namespace oneapi
#endif // _MKL_BLAS_CUBLAS_HELPER_HPP_
