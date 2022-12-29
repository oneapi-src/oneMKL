/*******************************************************************************
* Copyright 2022 Codeplay Software Ltd.
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

#define DOM_REAL      dft::detail::domain::REAL
#define DOM_COMPLEX   dft::detail::domain::COMPLEX
#define PREC_F        dft::detail::precision::SINGLE
#define PREC_D        dft::detail::precision::DOUBLE
#define DESC_RF       dft::detail::descriptor<PREC_F, DOM_REAL>
#define DESC_CF       dft::detail::descriptor<PREC_F, DOM_COMPLEX>
#define DESC_RD       dft::detail::descriptor<PREC_D, DOM_REAL>
#define DESC_CD       dft::detail::descriptor<PREC_D, DOM_COMPLEX>
#define DEPENDS_VEC_T const std::vector<sycl::event> &

#define ONEMKL_DFT_BACKWARD_INSTANTIATIONS(DESCRIPTOR_T, REAL_T, FORWARD_T, BACKWARD_T)         \
    /* Buffer API */                                                                            \
    template ONEMKL_EXPORT void compute_backward<DESCRIPTOR_T, REAL_T>(DESCRIPTOR_T & desc,     \
                                                                       sycl::buffer<REAL_T> &); \
    template ONEMKL_EXPORT void compute_backward<DESCRIPTOR_T, BACKWARD_T>(                     \
        DESCRIPTOR_T & desc, sycl::buffer<BACKWARD_T> &);                                       \
    template ONEMKL_EXPORT void compute_backward<DESCRIPTOR_T, BACKWARD_T, FORWARD_T>(          \
        DESCRIPTOR_T & desc, sycl::buffer<BACKWARD_T> &, sycl::buffer<FORWARD_T> &);            \
    template ONEMKL_EXPORT void compute_backward<DESCRIPTOR_T, REAL_T>(                         \
        DESCRIPTOR_T & desc, sycl::buffer<REAL_T> &, sycl::buffer<REAL_T> &);                   \
    template ONEMKL_EXPORT void compute_backward<DESCRIPTOR_T, REAL_T, REAL_T>(                 \
        DESCRIPTOR_T & desc, sycl::buffer<REAL_T> &, sycl::buffer<REAL_T> &);                   \
    template ONEMKL_EXPORT void compute_backward<DESCRIPTOR_T, REAL_T, REAL_T>(                 \
        DESCRIPTOR_T & desc, sycl::buffer<REAL_T> &, sycl::buffer<REAL_T> &,                    \
        sycl::buffer<REAL_T> &, sycl::buffer<REAL_T> &);                                        \
                                                                                                \
    /* USM API */                                                                               \
    template ONEMKL_EXPORT sycl::event compute_backward<DESCRIPTOR_T, REAL_T>(                  \
        DESCRIPTOR_T & desc, REAL_T *, DEPENDS_VEC_T);                                          \
    template ONEMKL_EXPORT sycl::event compute_backward<DESCRIPTOR_T, BACKWARD_T>(              \
        DESCRIPTOR_T & desc, BACKWARD_T *, DEPENDS_VEC_T);                                      \
    template ONEMKL_EXPORT sycl::event compute_backward<DESCRIPTOR_T, BACKWARD_T, FORWARD_T>(   \
        DESCRIPTOR_T & desc, BACKWARD_T *, FORWARD_T *, DEPENDS_VEC_T);                         \
    template ONEMKL_EXPORT sycl::event compute_backward<DESCRIPTOR_T, REAL_T>(                  \
        DESCRIPTOR_T & desc, REAL_T *, REAL_T *, DEPENDS_VEC_T);                                \
    template ONEMKL_EXPORT sycl::event compute_backward<DESCRIPTOR_T, REAL_T, REAL_T>(          \
        DESCRIPTOR_T & desc, REAL_T *, REAL_T *, DEPENDS_VEC_T);                                \
    template ONEMKL_EXPORT sycl::event compute_backward<DESCRIPTOR_T, REAL_T, REAL_T>(          \
        DESCRIPTOR_T & desc, REAL_T *, REAL_T *, REAL_T *, REAL_T *, DEPENDS_VEC_T);

#define ONEMKL_DFT_BACKWARD_INSTANTIATIONS_REAL_ONLY(DESCRIPTOR_T, COMPLEX_T)                \
    /* Buffer API */                                                                         \
    template ONEMKL_EXPORT void compute_backward<DESCRIPTOR_T, COMPLEX_T, COMPLEX_T>(        \
        DESCRIPTOR_T & desc, sycl::buffer<COMPLEX_T> &, sycl::buffer<COMPLEX_T> &);          \
    /* USM API */                                                                            \
    template ONEMKL_EXPORT sycl::event compute_backward<DESCRIPTOR_T, COMPLEX_T, COMPLEX_T>( \
        DESCRIPTOR_T & desc, COMPLEX_T *, COMPLEX_T *, DEPENDS_VEC_T);

ONEMKL_DFT_BACKWARD_INSTANTIATIONS(DESC_RF, float, float, std::complex<float>)
ONEMKL_DFT_BACKWARD_INSTANTIATIONS_REAL_ONLY(DESC_RF, std::complex<float>)
ONEMKL_DFT_BACKWARD_INSTANTIATIONS(DESC_CF, float, std::complex<float>, std::complex<float>)
ONEMKL_DFT_BACKWARD_INSTANTIATIONS(DESC_RD, double, double, std::complex<double>)
ONEMKL_DFT_BACKWARD_INSTANTIATIONS_REAL_ONLY(DESC_RD, std::complex<double>)
ONEMKL_DFT_BACKWARD_INSTANTIATIONS(DESC_CD, double, std::complex<double>, std::complex<double>)

#undef ONEMKL_DFT_BACKWARD_INSTANTIATIONS
#undef DOM_REAL
#undef DOM_COMPLEX
#undef PREC_F32
#undef PREC_F64
#undef DESC_RF32
#undef DESC_CF32
#undef DESC_RF64
#undef DESC_CF64
#undef DEPENDS_VEC_T
