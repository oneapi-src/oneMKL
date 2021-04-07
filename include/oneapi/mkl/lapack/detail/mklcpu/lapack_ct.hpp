/*******************************************************************************
* Copyright 2020-2021 Intel Corporation
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

#ifndef _ONEMKL_MKLCPU_LAPACK_CT_HPP_
#define _ONEMKL_MKLCPU_LAPACK_CT_HPP_

#include <CL/sycl.hpp>
#include <complex>
#include <cstdint>

#include "oneapi/mkl/types.hpp"
#include "oneapi/mkl/lapack/lapack_types.hpp"
#include "oneapi/mkl/detail/backend_selector.hpp"
#include "oneapi/mkl/lapack/detail/mklcpu/onemkl_lapack_mklcpu.hpp"

namespace oneapi {
namespace mkl {
namespace lapack {

#define LAPACK_BACKEND mklcpu
#include "oneapi/mkl/lapack/detail/mkl_common/lapack_ct.hxx"
#undef LAPACK_BACKEND

} //namespace lapack
} //namespace mkl
} //namespace oneapi

#endif //_ONEMKL_MKLCPU_LAPACK_CT_HPP_
