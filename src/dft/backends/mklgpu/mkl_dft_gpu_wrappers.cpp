/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#include "oneapi/mkl/dft/detail/mklgpu/onemkl_dft_mklgpu.hpp"
#include "dft/function_table.hpp"

#define WRAPPER_VERSION 1

extern "C" dft_function_table_t mkl_dft_table = {
    WRAPPER_VERSION,
#define ONEAPI_MKL_DFT_BACKEND_SIGNATURES(EXT)                                    \
    oneapi::mkl::dft::mklgpu::commit_##EXT,                                       \
        oneapi::mkl::dft::mklgpu::compute_forward_buffer_inplace_##EXT,           \
        oneapi::mkl::dft::mklgpu::compute_forward_buffer_inplace_split_##EXT,     \
        oneapi::mkl::dft::mklgpu::compute_forward_buffer_outofplace_##EXT,        \
        oneapi::mkl::dft::mklgpu::compute_forward_buffer_outofplace_split_##EXT,  \
        oneapi::mkl::dft::mklgpu::compute_forward_usm_inplace_##EXT,              \
        oneapi::mkl::dft::mklgpu::compute_forward_usm_inplace_split_##EXT,        \
        oneapi::mkl::dft::mklgpu::compute_forward_usm_outofplace_##EXT,           \
        oneapi::mkl::dft::mklgpu::compute_forward_usm_outofplace_split_##EXT,     \
        oneapi::mkl::dft::mklgpu::compute_backward_buffer_inplace_##EXT,          \
        oneapi::mkl::dft::mklgpu::compute_backward_buffer_inplace_split_##EXT,    \
        oneapi::mkl::dft::mklgpu::compute_backward_buffer_outofplace_##EXT,       \
        oneapi::mkl::dft::mklgpu::compute_backward_buffer_outofplace_split_##EXT, \
        oneapi::mkl::dft::mklgpu::compute_backward_usm_inplace_##EXT,             \
        oneapi::mkl::dft::mklgpu::compute_backward_usm_inplace_split_##EXT,       \
        oneapi::mkl::dft::mklgpu::compute_backward_usm_outofplace_##EXT,          \
        oneapi::mkl::dft::mklgpu::compute_backward_usm_outofplace_split_##EXT

    ONEAPI_MKL_DFT_BACKEND_SIGNATURES(f), ONEAPI_MKL_DFT_BACKEND_SIGNATURES(c),
    ONEAPI_MKL_DFT_BACKEND_SIGNATURES(d), ONEAPI_MKL_DFT_BACKEND_SIGNATURES(z)

#undef ONEAPI_MKL_DFT_BACKEND_SIGNATURES
};
