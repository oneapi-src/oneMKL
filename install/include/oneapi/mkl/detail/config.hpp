/*******************************************************************************
* Copyright 2020-2022 Intel Corporation
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

#ifndef ONEMKL_CONFIG_H
#define ONEMKL_CONFIG_H

/* #undef ENABLE_CUBLAS_BACKEND */
/* #undef ENABLE_CUFFT_BACKEND */
/* #undef ENABLE_CURAND_BACKEND */
/* #undef ENABLE_CUSOLVER_BACKEND */
#define ENABLE_MKLCPU_BACKEND
#define ENABLE_MKLGPU_BACKEND
/* #undef ENABLE_NETLIB_BACKEND */
/* #undef ENABLE_ROCBLAS_BACKEND */
/* #undef ENABLE_ROCFFT_BACKEND */
/* #undef ENABLE_ROCRAND_BACKEND */
/* #undef ENABLE_ROCSOLVER_BACKEND */
/* #undef ENABLE_PORTBLAS_BACKEND */
/* #undef ENABLE_PORTBLAS_BACKEND_AMD_GPU */
/* #undef ENABLE_PORTBLAS_BACKEND_INTEL_CPU */
/* #undef ENABLE_PORTBLAS_BACKEND_INTEL_GPU */
/* #undef ENABLE_PORTBLAS_BACKEND_NVIDIA_GPU */
#define BUILD_SHARED_LIBS
/* #undef REF_BLAS_LIBNAME */
/* #undef REF_CBLAS_LIBNAME */

#endif
