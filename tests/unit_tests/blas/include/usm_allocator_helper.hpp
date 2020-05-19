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

#ifndef __USM_ALLOCATOR_HELPER_HPP
#define __USM_ALLOCATOR_HELPER_HPP

#include <stdlib.h>
#include <cstddef>
#include <limits>
#include <type_traits>
#include "test_helper.hpp"

template <typename T, int align>
class usm_allocator_helper : public cl::sycl::usm_allocator<T, cl::sycl::usm::alloc::shared, 64> {
    cl::sycl::context c;
    cl::sycl::device d;

public:
    template <typename U>
    struct rebind {
        typedef usm_allocator_helper<U, align> other;
    };

    template <typename U, int align2>
    usm_allocator_helper(usm_allocator_helper<U, align2> &other) noexcept {
        other.c = c;
        other.d = d;
    }
    template <typename U, int align2>
    usm_allocator_helper(usm_allocator_helper<U, align2> &&other) noexcept {
        other.c = c;
        other.d = d;
    }

    T *allocate(size_t n) {
        void *mem = onemkl::malloc_shared(align, n * sizeof(T), d, c);
        if (!mem)
            throw std::bad_alloc();

        return static_cast<T *>(mem);
    }

    void deallocate(T *p, size_t n) noexcept {
        onemkl::free_shared(p, c);
    }

    usm_allocator_helper(cl::sycl::context cxt, cl::sycl::device dev) noexcept
            : cl::sycl::usm_allocator<T, cl::sycl::usm::alloc::shared, align>(cxt, dev),
              c(cxt),
              d(dev){};
};

#endif
