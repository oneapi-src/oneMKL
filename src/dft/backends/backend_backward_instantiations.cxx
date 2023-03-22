/*******************************************************************************
* Copyright Codeplay Software Ltd.
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

using desc_rf_t =
    dft::detail::descriptor<dft::detail::precision::SINGLE, dft::detail::domain::REAL>;
using desc_cf_t =
    dft::detail::descriptor<dft::detail::precision::SINGLE, dft::detail::domain::COMPLEX>;
using desc_rd_t =
    dft::detail::descriptor<dft::detail::precision::DOUBLE, dft::detail::domain::REAL>;
using desc_cd_t =
    dft::detail::descriptor<dft::detail::precision::DOUBLE, dft::detail::domain::COMPLEX>;
using depends_vec_t = const std::vector<sycl::event> &;

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
        DESCRIPTOR_T & desc, REAL_T *, depends_vec_t);                                          \
    template ONEMKL_EXPORT sycl::event compute_backward<DESCRIPTOR_T, BACKWARD_T>(              \
        DESCRIPTOR_T & desc, BACKWARD_T *, depends_vec_t);                                      \
    template ONEMKL_EXPORT sycl::event compute_backward<DESCRIPTOR_T, BACKWARD_T, FORWARD_T>(   \
        DESCRIPTOR_T & desc, BACKWARD_T *, FORWARD_T *, depends_vec_t);                         \
    template ONEMKL_EXPORT sycl::event compute_backward<DESCRIPTOR_T, REAL_T>(                  \
        DESCRIPTOR_T & desc, REAL_T *, REAL_T *, depends_vec_t);                                \
    template ONEMKL_EXPORT sycl::event compute_backward<DESCRIPTOR_T, REAL_T, REAL_T>(          \
        DESCRIPTOR_T & desc, REAL_T *, REAL_T *, depends_vec_t);                                \
    template ONEMKL_EXPORT sycl::event compute_backward<DESCRIPTOR_T, REAL_T, REAL_T>(          \
        DESCRIPTOR_T & desc, REAL_T *, REAL_T *, REAL_T *, REAL_T *, depends_vec_t);

#define ONEMKL_DFT_BACKWARD_INSTANTIATIONS_REAL_ONLY(DESCRIPTOR_T, COMPLEX_T)                \
    /* Buffer API */                                                                         \
    template ONEMKL_EXPORT void compute_backward<DESCRIPTOR_T, COMPLEX_T, COMPLEX_T>(        \
        DESCRIPTOR_T & desc, sycl::buffer<COMPLEX_T> &, sycl::buffer<COMPLEX_T> &);          \
    /* USM API */                                                                            \
    template ONEMKL_EXPORT sycl::event compute_backward<DESCRIPTOR_T, COMPLEX_T, COMPLEX_T>( \
        DESCRIPTOR_T & desc, COMPLEX_T *, COMPLEX_T *, depends_vec_t);

ONEMKL_DFT_BACKWARD_INSTANTIATIONS(desc_rf_t, float, float, std::complex<float>)
ONEMKL_DFT_BACKWARD_INSTANTIATIONS_REAL_ONLY(desc_rf_t, std::complex<float>)
ONEMKL_DFT_BACKWARD_INSTANTIATIONS(desc_cf_t, float, std::complex<float>, std::complex<float>)
ONEMKL_DFT_BACKWARD_INSTANTIATIONS(desc_rd_t, double, double, std::complex<double>)
ONEMKL_DFT_BACKWARD_INSTANTIATIONS_REAL_ONLY(desc_rd_t, std::complex<double>)
ONEMKL_DFT_BACKWARD_INSTANTIATIONS(desc_cd_t, double, std::complex<double>, std::complex<double>)

#undef ONEMKL_DFT_BACKWARD_INSTANTIATIONS
#undef ONEMKL_DFT_BACKWARD_INSTANTIATIONS_REAL_ONLY
