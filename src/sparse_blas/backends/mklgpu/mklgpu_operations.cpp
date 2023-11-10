/*******************************************************************************
* Copyright 2023 Codeplay Software Ltd.
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

#include "../mkl_common/mkl_helper.hpp"

#include "oneapi/mkl/sparse_blas/detail/mklgpu/onemkl_sparse_blas_mklgpu.hpp"

namespace oneapi::mkl::sparse::mklgpu {

#include "../mkl_common/mkl_operations.cxx"

} // namespace oneapi::mkl::sparse::mklgpu
