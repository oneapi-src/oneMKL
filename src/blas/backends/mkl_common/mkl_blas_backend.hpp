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

#pragma once

#include <complex>

// INTEL_MKL_VERSION
#include "mkl_version.h"
#include "oneapi/mkl/types.hpp"

namespace oneapi {
namespace mkl {
namespace blas {

namespace column_major {

#include "mkl_blas_backend.hxx"

}

namespace row_major {

#include "mkl_blas_backend.hxx"

}

} // namespace blas
} // namespace mkl
} // namespace oneapi
