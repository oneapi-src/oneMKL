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

#ifndef _ONEMKL_BLAS_LOADER_HPP_
#define _ONEMKL_BLAS_LOADER_HPP_

#include <complex>
#include <cstdint>
#include <CL/sycl.hpp>

#include "oneapi/mkl/types.hpp"

#include "oneapi/mkl/detail/export.hpp"
#include "oneapi/mkl/detail/get_device_id.hpp"

namespace oneapi {
namespace mkl {
namespace blas {
namespace column_major {
namespace detail {

#include "blas_loader.hxx"

} //namespace detail
} //namespace column_major
namespace row_major {
namespace detail {

#include "blas_loader.hxx"

} //namespace detail
} //namespace row_major
} //namespace blas
} //namespace mkl
} //namespace oneapi

#endif //_ONEMKL_BLAS_LOADER_HPP_
