/*******************************************************************************
* Copyright Codeplay Software
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

#ifndef _SYCLBLAS_COMMON_HPP_
#define _SYCLBLAS_COMMON_HPP_

#include "sycl_blas.hpp"
#include "oneapi/mkl/types.hpp"
#include "oneapi/mkl/exceptions.hpp"

#include <tuple>
#include <utility>

namespace oneapi {
namespace mkl {
namespace blas {
namespace syclblas {

namespace detail {
// SYCL-BLAS handle type. Constructed with sycl::queue.
using handle_t = ::blas::SB_Handle;

// SYCL-BLAS buffer iterator. Constructed with sycl::buffer<ElemT,1>
template <typename ElemT>
using buffer_iterator_t = ::blas::BufferIterator<ElemT>;

/** A trait for obtaining equivalent SYCL-BLAS API types from oneMKL API
 *  types.
 * 
 *  @tparam InputT is the oneMKL type.
 *  syclblas_type<InputT>::type should be the equivalent SYCL-BLAS type.
**/
template <typename InputT>
struct syclblas_type;

#define DEF_SYCLBLAS_TYPE(onemkl_t, syclblas_t) \
    template <>                                 \
    struct syclblas_type<onemkl_t> {            \
        using type = syclblas_t;                \
    };

DEF_SYCLBLAS_TYPE(sycl::queue, handle_t)
DEF_SYCLBLAS_TYPE(int64_t, int64_t)
DEF_SYCLBLAS_TYPE(sycl::half, sycl::half)
DEF_SYCLBLAS_TYPE(float, float)
DEF_SYCLBLAS_TYPE(double, double)
DEF_SYCLBLAS_TYPE(oneapi::mkl::transpose, char)
DEF_SYCLBLAS_TYPE(oneapi::mkl::uplo, char)
DEF_SYCLBLAS_TYPE(oneapi::mkl::side, char)
DEF_SYCLBLAS_TYPE(oneapi::mkl::diag, char)
// Passthrough of SYCL-BLAS arg types for more complex wrapping.
DEF_SYCLBLAS_TYPE(::blas::gemm_batch_type_t, ::blas::gemm_batch_type_t)

#undef DEF_SYCLBLAS_TYPE

template <typename ElemT>
struct syclblas_type<sycl::buffer<ElemT, 1>> {
    using type = buffer_iterator_t<ElemT>;
};

/** Convert a OneMKL argument to the type required for SYCL-BLAS.
 *  
 *  @tparam InputT The OneMKL type.
 *  @param input The value of the oneMKL type.
 *  @return The SYCL-BLAS value with appropriate type.
**/
template <typename InputT>
inline typename syclblas_type<InputT>::type convert_to_syclblas_type(InputT& input) {
    return typename syclblas_type<InputT>::type(input);
}

template <>
inline char convert_to_syclblas_type<oneapi::mkl::transpose>(oneapi::mkl::transpose& trans) {
    if (trans == oneapi::mkl::transpose::nontrans) {
        return 'n';
    }
    else if (trans == oneapi::mkl::transpose::trans) {
        return 't';
    }
    else { // trans == oneapi::mkl::transpose::conjtrans
        return 'c';
    }
}

template <>
inline char convert_to_syclblas_type<oneapi::mkl::uplo>(oneapi::mkl::uplo& upper_lower) {
    if (upper_lower == oneapi::mkl::uplo::upper) {
        return 'u';
    }
    else {
        return 'l';
    }
}

template <>
inline char convert_to_syclblas_type<oneapi::mkl::side>(oneapi::mkl::side& left_right) {
    if (left_right == oneapi::mkl::side::left) {
        return 'l';
    }
    else {
        return 'r';
    }
}

template <>
inline char convert_to_syclblas_type<oneapi::mkl::diag>(oneapi::mkl::diag& unit_diag) {
    if (unit_diag == oneapi::mkl::diag::unit) {
        return 'u';
    }
    else {
        return 'n';
    }
}

template <typename... ArgT>
inline auto convert_to_syclblas_type(ArgT... args) {
    return std::make_tuple(convert_to_syclblas_type(args)...);
}

/** Throw an MKL unsuppored device exception if a certain argument
 *  type is found in the argument pack.
 *  
 *  @tparam CheckT is type to look for a template parameter pack.
 *  @tparam AspectVal is the device aspect required to support CheckT.
**/
template <typename CheckT, sycl::aspect AspectVal>
struct throw_if_unsupported_by_device {
    /** Operator to throw if unsupported.
 * 
 *  @tparam ArgTs The argument types to check.
 *  @param The message to include in the exception.
 *  @param q is the sycl::queue.
 *  @param args are the remaining args to check for CheckT in.
**/
    template <typename... ArgTs>
    void operator()(const std::string& message, sycl::queue q, ArgTs... args) {
        static constexpr bool checkTypeInPack = (std::is_same_v<CheckT, ArgTs> || ...);
        if (checkTypeInPack) {
            if (!q.get_info<sycl::info::queue::device>().has(AspectVal)) {
                throw mkl::unsupported_device("blas", message,
                                              q.get_info<sycl::info::queue::device>());
            }
        }
    }
};

} // namespace detail

#define CALL_SYCLBLAS_FN(syclblasFunc, ...)                                                     \
    if constexpr (is_column_major()) {                                                          \
        detail::throw_if_unsupported_by_device<sycl::buffer<double>, sycl::aspect::fp64>{}(     \
            " SYCL-BLAS function requiring fp64 support", __VA_ARGS__);                         \
        detail::throw_if_unsupported_by_device<sycl::buffer<sycl::half>, sycl::aspect::fp16>{}( \
            " SYCL-BLAS function requiring fp16 support", __VA_ARGS__);                         \
        auto args = detail::convert_to_syclblas_type(__VA_ARGS__);                              \
        auto fn = [](auto&&... targs) {                                                         \
            syclblasFunc(std::forward<decltype(targs)>(targs)...);                              \
        };                                                                                      \
        std::apply(fn, args);                                                                   \
    }                                                                                           \
    else {                                                                                      \
        throw unimplemented("blas", "SyclBLAS function", " for row-major");                     \
    }

} // namespace syclblas
} // namespace blas
} // namespace mkl
} // namespace oneapi

#endif // _SYCLBLAS_COMMON_HPP_
