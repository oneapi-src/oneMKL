/***************************************************************************
*  Copyright (C) Codeplay Software Limited
*  Copyright (C) 2022 Heidelberg University, Engineering Mathematics and Computing Lab (EMCL) and Computing Centre (URZ)
*
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
 * @file rocblas*.cpp : contains the implementation of all the routines
 * for rocBLAS backend
 */
#ifndef _ROCBLAS_HELPER_HPP_

#define _ROCBLAS_HELPER_HPP_

#include <rocblas.h>
#include <complex>
#include "oneapi/mkl/types.hpp"

namespace oneapi {
namespace mkl {
namespace blas {
namespace rocblas {

// The static assert to make sure that all index types used in
// src/oneMKL/backend/rocblas/blas.hpp interface are int64_t
template <typename... Next>
struct is_int64 : std::false_type {};

template <typename First>
struct is_int64<First> : std::is_same<int64_t, First> {};

template <typename First, typename... Next>
struct is_int64<First, Next...>
        : std::integral_constant<bool, std::is_same<int64_t, First>::value &&
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
                "Rocblas index overflow. rocblas does not support 64 bit integer as "
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

class rocblas_error : virtual public std::runtime_error {
protected:
    inline const char *rocblas_error_map(rocblas_status error) {
        switch (error) {
            case rocblas_status_success: return "rocblas_status_success";
            case rocblas_status_invalid_handle: return "rocblas_status_invalid_handle";
            case rocblas_status_not_implemented: return "rocblas_status_not_implemented";
            case rocblas_status_invalid_pointer: return "rocblas_status_invalid_pointer";
            case rocblas_status_invalid_size: return "rocblas_status_invalid_size";
            case rocblas_status_memory_error: return "rocblas_status_memory_error";
            case rocblas_status_internal_error: return "rocblas_status_internal_error";
            case rocblas_status_perf_degraded: return "rocblas_status_perf_degraded";
            case rocblas_status_size_query_mismatch: return "rocblas_status_size_query_mismatch";
            case rocblas_status_size_increased: return "rocblas_status_size_increased";
            case rocblas_status_size_unchanged: return "rocblas_status_size_unchanged";
            case rocblas_status_invalid_value: return "rocblas_status_invalid_value";
            case rocblas_status_continue: return "rocblas_status_continue";
            case rocblas_status_check_numerics_fail: return "rocblas_status_check_numerics_fail";

            default: return "<unknown>";
        }
    }

    int error_number; ///< Error number
public:
    /** Constructor (C++ STL string, rocblas_status).
   *  @param msg The error message
   *  @param err_num error number
   */
    explicit rocblas_error(std::string message, rocblas_status result)
            : std::runtime_error((message + std::string(rocblas_error_map(result)))) {
        error_number = static_cast<int>(result);
    }

    /** Destructor.
   *  Virtual to allow for subclassing.
   */
    virtual ~rocblas_error() throw() {}

    /** Returns error number.
   *  @return #error_number
   */
    virtual int getErrorNumber() const throw() {
        return error_number;
    }
};

class hip_error : virtual public std::runtime_error {
protected:
    inline const char *hip_error_map(hipError_t result) {
        return hipGetErrorName(result);
    }
    int error_number; ///< error number
public:
    /** Constructor (C++ STL string, hipError_t).
   *  @param msg The error message
   *  @param err_num Error number
   */
    explicit hip_error(std::string message, hipError_t result)
            : std::runtime_error((message + std::string(hip_error_map(result)))) {
        error_number = static_cast<int>(result);
    }

    /** Destructor.
   *  Virtual to allow for subclassing.
   */
    virtual ~hip_error() throw() {}

    /** Returns error number.
   *  @return #error_number
   */
    virtual int getErrorNumber() const throw() {
        return error_number;
    }
};

#define HIP_ERROR_FUNC(name, err, ...)                                 \
    err = name(__VA_ARGS__);                                           \
    if (err != HIP_SUCCESS) {                                          \
        throw hip_error(std::string(#name) + std::string(" : "), err); \
    }

#define ROCBLAS_ERROR_FUNC(name, err, ...)                                 \
    err = name(__VA_ARGS__);                                               \
    if (err != rocblas_status_success) {                                   \
        throw rocblas_error(std::string(#name) + std::string(" : "), err); \
    }

inline rocblas_operation get_rocblas_operation(oneapi::mkl::transpose trn) {
    switch (trn) {
        case oneapi::mkl::transpose::nontrans: return rocblas_operation_none;
        case oneapi::mkl::transpose::trans: return rocblas_operation_transpose;
        case oneapi::mkl::transpose::conjtrans: return rocblas_operation_conjugate_transpose;
        default: throw "Wrong transpose Operation.";
    }
}

inline rocblas_fill get_rocblas_fill_mode(oneapi::mkl::uplo ul) {
    switch (ul) {
        case oneapi::mkl::uplo::upper: return rocblas_fill_upper;
        case oneapi::mkl::uplo::lower: return rocblas_fill_lower;
        default: throw "Wrong fill mode.";
    }
}

inline rocblas_diagonal get_rocblas_diag_type(oneapi::mkl::diag un) {
    switch (un) {
        case oneapi::mkl::diag::unit: return rocblas_diagonal_unit;
        case oneapi::mkl::diag::nonunit: return rocblas_diagonal_non_unit;
        default: throw "Wrong diag type.";
    }
}

inline rocblas_side get_rocblas_side_mode(oneapi::mkl::side lr) {
    switch (lr) {
        case oneapi::mkl::side::left: return rocblas_side_left;
        case oneapi::mkl::side::right: return rocblas_side_right;
        default: throw "Wrong side mode.";
    }
}

/*converting std::complex<T> to roc_<T>_complex 
             sycl::half      to rocblas_half*/
template <typename T>
struct RocEquivalentType {
    using Type = T;
};

template <>
struct RocEquivalentType<std::complex<float>> {
    using Type = rocblas_float_complex;
};
template <>
struct RocEquivalentType<std::complex<double>> {
    using Type = rocblas_double_complex;
};
template <>
struct RocEquivalentType<sycl::half> {
    using Type = rocblas_half;
};

} // namespace rocblas
} // namespace blas
} // namespace mkl
} // namespace oneapi
#endif // _ROCBLAS_HELPER_HPP_
