/***************************************************************************
*  Copyright (C) Codeplay Software Limited
*  Licensed under the Apache License, Version 2.0 (the "License");
*  you may not use this file except in compliance with the License.
*  You may obtain a copy of the License at
*
*      http://www.apache.org/licenses/LICENSE-2.0
*
*  For your convenience, a copy of the License has been included in this
*  repository.
*
*  Unless required by applicable law or agreed to in writing, software
*  distributed under the License is distributed on an "AS IS" BASIS,
*  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*  See the License for the specific language governing permissions and
*  limitations under the License.
*
**************************************************************************/

#ifndef _ONEMKL_SPARSE_BLAS_BACKENDS_ROCSPARSE_TASK_HPP_
#define _ONEMKL_SPARSE_BLAS_BACKENDS_ROCSPARSE_TASK_HPP_

#include "rocsparse_handles.hpp"
#include "rocsparse_scope_handle.hpp"

#define BACKEND                      rocsparse
#define BACKEND_SCOPE_CONTEXT_HANDLE RocsparseScopedContextHandler
#include "sparse_blas/backends/common_launch_task.hxx"
#undef BACKEND
#undef BACKEND_SCOPE_CONTEXT_HANDLE

#endif // _ONEMKL_SPARSE_BLAS_BACKENDS_ROCSPARSE_TASK_HPP_