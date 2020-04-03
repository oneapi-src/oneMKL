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

#ifndef _LOADER_HPP_
#define _LOADER_HPP_

#include <cstdint>
#include <map>
#include "blas/function_table.hpp"

#define SPEC_VERSION 1

#ifdef __linux__
    #include <dlfcn.h>
    #define LIB_TYPE                 void *
    #define GET_LIB_HANDLE(libname)  dlopen((libname), RTLD_LAZY | RTLD_GLOBAL)
    #define GET_FUNC(lib, fn)        dlsym(lib, (fn))
    #define FREE_LIB_HANDLE(libname) dlclose(libname)
    #define ERROR_MSG                dlerror()
#endif

namespace onemkl {
namespace blas {
namespace detail {

class table_initializer {
    struct handle_deleter {
        using pointer = LIB_TYPE;
        void operator()(pointer p) const {
            ::FREE_LIB_HANDLE(p);
        }
    };
    using dlhandle = std::unique_ptr<LIB_TYPE, handle_deleter>;

public:
    function_table_t &operator[](const char *libname) {
        auto lib = tables.find(libname);
        if (lib != tables.end())
            return lib->second;
        return add_table(libname);
    }

private:
    function_table_t &add_table(const char *libname) {
        auto handle = dlhandle{ ::GET_LIB_HANDLE(libname) };
        if (!handle) {
            std::cerr << ERROR_MSG << '\n';
            throw std::runtime_error{ "Couldn't load selected backend" };
        }

        auto t = reinterpret_cast<function_table_t *>(::GET_FUNC(handle.get(), "mkl_blas_table"));

        if (!t) {
            std::cerr << ERROR_MSG << '\n';
            throw std::runtime_error{ "Couldn't load functions from selected backend" };
        }
        if (t->version != SPEC_VERSION)
            throw std::runtime_error{ "Loaded oneMKL specification version mismatch" };

        handles[libname] = std::move(handle);
        tables[libname]  = *t;
        return *t;
    }

    std::map<const char *, function_table_t> tables;
    std::map<const char *, dlhandle> handles;
};

static table_initializer function_tables;

} //namespace detail
} // namespace blas
} // namespace onemkl

#endif //_LOADER_HPP_
