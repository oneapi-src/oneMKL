/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include "blas/function_table.hpp"
#include "onemkl/blas/detail/mklcpu/onemkl_blas_mklcpu.hpp"

#define WRAPPER_VERSION 1

extern "C" ONEMKL_EXPORT function_table_t mkl_blas_table = {
    WRAPPER_VERSION,
    onemkl::mklcpu::asum,
    onemkl::mklcpu::asum,
    onemkl::mklcpu::asum,
    onemkl::mklcpu::asum,
    onemkl::mklcpu::axpy,
    onemkl::mklcpu::axpy,
    onemkl::mklcpu::axpy,
    onemkl::mklcpu::axpy,
    onemkl::mklcpu::copy,
    onemkl::mklcpu::copy,
    onemkl::mklcpu::copy,
    onemkl::mklcpu::copy,
    onemkl::mklcpu::dot,
    onemkl::mklcpu::dot,
    onemkl::mklcpu::dot,
    onemkl::mklcpu::dotc,
    onemkl::mklcpu::dotc,
    onemkl::mklcpu::dotu,
    onemkl::mklcpu::dotu,
    onemkl::mklcpu::iamin,
    onemkl::mklcpu::iamin,
    onemkl::mklcpu::iamin,
    onemkl::mklcpu::iamin,
    onemkl::mklcpu::iamax,
    onemkl::mklcpu::iamax,
    onemkl::mklcpu::iamax,
    onemkl::mklcpu::iamax,
    onemkl::mklcpu::nrm2,
    onemkl::mklcpu::nrm2,
    onemkl::mklcpu::nrm2,
    onemkl::mklcpu::nrm2,
    onemkl::mklcpu::rot,
    onemkl::mklcpu::rot,
    onemkl::mklcpu::rot,
    onemkl::mklcpu::rot,
    onemkl::mklcpu::rotg,
    onemkl::mklcpu::rotg,
    onemkl::mklcpu::rotg,
    onemkl::mklcpu::rotg,
    onemkl::mklcpu::rotm,
    onemkl::mklcpu::rotm,
    onemkl::mklcpu::rotmg,
    onemkl::mklcpu::rotmg,
    onemkl::mklcpu::scal,
    onemkl::mklcpu::scal,
    onemkl::mklcpu::scal,
    onemkl::mklcpu::scal,
    onemkl::mklcpu::scal,
    onemkl::mklcpu::scal,
    onemkl::mklcpu::sdsdot,
    onemkl::mklcpu::swap,
    onemkl::mklcpu::swap,
    onemkl::mklcpu::swap,
    onemkl::mklcpu::swap,
    onemkl::mklcpu::gbmv,
    onemkl::mklcpu::gbmv,
    onemkl::mklcpu::gbmv,
    onemkl::mklcpu::gbmv,
    onemkl::mklcpu::gemv,
    onemkl::mklcpu::gemv,
    onemkl::mklcpu::gemv,
    onemkl::mklcpu::gemv,
    onemkl::mklcpu::ger,
    onemkl::mklcpu::ger,
    onemkl::mklcpu::gerc,
    onemkl::mklcpu::gerc,
    onemkl::mklcpu::geru,
    onemkl::mklcpu::geru,
    onemkl::mklcpu::hbmv,
    onemkl::mklcpu::hbmv,
    onemkl::mklcpu::hemv,
    onemkl::mklcpu::hemv,
    onemkl::mklcpu::her,
    onemkl::mklcpu::her,
    onemkl::mklcpu::her2,
    onemkl::mklcpu::her2,
    onemkl::mklcpu::hpmv,
    onemkl::mklcpu::hpmv,
    onemkl::mklcpu::hpr,
    onemkl::mklcpu::hpr,
    onemkl::mklcpu::hpr2,
    onemkl::mklcpu::hpr2,
    onemkl::mklcpu::sbmv,
    onemkl::mklcpu::sbmv,
    onemkl::mklcpu::spmv,
    onemkl::mklcpu::spmv,
    onemkl::mklcpu::spr,
    onemkl::mklcpu::spr,
    onemkl::mklcpu::spr2,
    onemkl::mklcpu::spr2,
    onemkl::mklcpu::symv,
    onemkl::mklcpu::symv,
    onemkl::mklcpu::syr,
    onemkl::mklcpu::syr,
    onemkl::mklcpu::syr2,
    onemkl::mklcpu::syr2,
    onemkl::mklcpu::tbmv,
    onemkl::mklcpu::tbmv,
    onemkl::mklcpu::tbmv,
    onemkl::mklcpu::tbmv,
    onemkl::mklcpu::tbsv,
    onemkl::mklcpu::tbsv,
    onemkl::mklcpu::tbsv,
    onemkl::mklcpu::tbsv,
    onemkl::mklcpu::tpmv,
    onemkl::mklcpu::tpmv,
    onemkl::mklcpu::tpmv,
    onemkl::mklcpu::tpmv,
    onemkl::mklcpu::tpsv,
    onemkl::mklcpu::tpsv,
    onemkl::mklcpu::tpsv,
    onemkl::mklcpu::tpsv,
    onemkl::mklcpu::trmv,
    onemkl::mklcpu::trmv,
    onemkl::mklcpu::trmv,
    onemkl::mklcpu::trmv,
    onemkl::mklcpu::trsv,
    onemkl::mklcpu::trsv,
    onemkl::mklcpu::trsv,
    onemkl::mklcpu::trsv,
    onemkl::mklcpu::gemm,
    onemkl::mklcpu::gemm,
    onemkl::mklcpu::gemm,
    onemkl::mklcpu::gemm,
    onemkl::mklcpu::gemm,
    onemkl::mklcpu::hemm,
    onemkl::mklcpu::hemm,
    onemkl::mklcpu::herk,
    onemkl::mklcpu::herk,
    onemkl::mklcpu::her2k,
    onemkl::mklcpu::her2k,
    onemkl::mklcpu::symm,
    onemkl::mklcpu::symm,
    onemkl::mklcpu::symm,
    onemkl::mklcpu::symm,
    onemkl::mklcpu::syrk,
    onemkl::mklcpu::syrk,
    onemkl::mklcpu::syrk,
    onemkl::mklcpu::syrk,
    onemkl::mklcpu::syr2k,
    onemkl::mklcpu::syr2k,
    onemkl::mklcpu::syr2k,
    onemkl::mklcpu::syr2k,
    onemkl::mklcpu::trmm,
    onemkl::mklcpu::trmm,
    onemkl::mklcpu::trmm,
    onemkl::mklcpu::trmm,
    onemkl::mklcpu::trsm,
    onemkl::mklcpu::trsm,
    onemkl::mklcpu::trsm,
    onemkl::mklcpu::trsm,
    onemkl::mklcpu::gemm_batch,
    onemkl::mklcpu::gemm_batch,
    onemkl::mklcpu::gemm_batch,
    onemkl::mklcpu::gemm_batch,
    onemkl::mklcpu::gemm_batch,
    onemkl::mklcpu::gemm_batch,
    onemkl::mklcpu::gemm_batch,
    onemkl::mklcpu::gemm_batch,
    onemkl::mklcpu::trsm_batch,
    onemkl::mklcpu::trsm_batch,
    onemkl::mklcpu::trsm_batch,
    onemkl::mklcpu::trsm_batch,
    onemkl::mklcpu::trsm_batch,
    onemkl::mklcpu::trsm_batch,
    onemkl::mklcpu::trsm_batch,
    onemkl::mklcpu::trsm_batch,
    onemkl::mklcpu::gemmt,
    onemkl::mklcpu::gemmt,
    onemkl::mklcpu::gemmt,
    onemkl::mklcpu::gemmt,
    onemkl::mklcpu::gemm_ext,
    onemkl::mklcpu::gemm_ext,
    onemkl::mklcpu::gemm_ext,
    onemkl::mklcpu::gemm_ext,
    onemkl::mklcpu::gemm_ext,
    onemkl::mklcpu::gemm_ext,
    onemkl::mklcpu::gemm_ext,
};
