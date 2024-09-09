/*******************************************************************************
* Copyright 2023 Codeplay Software Ltd.
*
* (*Licensed under the Apache License, Version 2.0 )(the "License");
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

#ifndef _ONEMKL_SPARSE_BLAS_MACROS_HPP_
#define _ONEMKL_SPARSE_BLAS_MACROS_HPP_

#define FOR_EACH_FP_TYPE(DEFINE_MACRO)      \
    DEFINE_MACRO(float, _rf);               \
    DEFINE_MACRO(double, _rd);              \
    DEFINE_MACRO(std::complex<float>, _cf); \
    DEFINE_MACRO(std::complex<double>, _cd)

#define FOR_EACH_FP_AND_INT_TYPE_HELPER(DEFINE_MACRO, INT_TYPE, INT_SUFFIX) \
    DEFINE_MACRO(float, _rf, INT_TYPE, INT_SUFFIX);                         \
    DEFINE_MACRO(double, _rd, INT_TYPE, INT_SUFFIX);                        \
    DEFINE_MACRO(std::complex<float>, _cf, INT_TYPE, INT_SUFFIX);           \
    DEFINE_MACRO(std::complex<double>, _cd, INT_TYPE, INT_SUFFIX)

#define FOR_EACH_FP_AND_INT_TYPE(DEFINE_MACRO)                         \
    FOR_EACH_FP_AND_INT_TYPE_HELPER(DEFINE_MACRO, std::int32_t, _i32); \
    FOR_EACH_FP_AND_INT_TYPE_HELPER(DEFINE_MACRO, std::int64_t, _i64)

#define THROW_IF_NULLPTR(FUNC_NAME, PTR)                                       \
    if (!(PTR)) {                                                              \
        throw mkl::uninitialized("sparse_blas", FUNC_NAME,                     \
                                 std::string(#PTR) + " must not be nullptr."); \
    }

#endif // _ONEMKL_SPARSE_BLAS_MACROS_HPP_
