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

#ifndef _LOADER_HPP_
#define _LOADER_HPP_

#include <cstdint>
#include <map>

#include "oneapi/mkl/detail/backends_table.hpp"
#include "oneapi/mkl/detail/exceptions.hpp"

#define SPEC_VERSION 1

#ifdef __linux__
#include <dlfcn.h>
#define LIB_TYPE                 void *
#define GET_LIB_HANDLE(libname)  dlopen((libname), RTLD_LAZY | RTLD_GLOBAL)
#define GET_FUNC(lib, fn)        dlsym(lib, (fn))
#define FREE_LIB_HANDLE(libname) dlclose(libname)
#define ERROR_MSG                dlerror()
#elif defined(_WIN64)
#include <windows.h>
#define LIB_TYPE                 HINSTANCE
#define GET_LIB_HANDLE(libname)  LoadLibrary(libname)
#define GET_FUNC(lib, fn)        GetProcAddress((lib), (fn))
#define FREE_LIB_HANDLE(libname) FreeLibrary(libname)
#define ERROR_MSG                GetLastErrorStdStr()
#endif

namespace oneapi {
namespace mkl {
namespace detail {

template <oneapi::mkl::domain domain_id, typename function_table_t>
class table_initializer {
    struct handle_deleter {
        using pointer = LIB_TYPE;
        void operator()(pointer p) const {
            ::FREE_LIB_HANDLE(p);
        }
    };
    using dlhandle = std::unique_ptr<LIB_TYPE, handle_deleter>;

public:
    function_table_t &operator[](oneapi::mkl::device key) {
        auto lib = tables.find(key);
        if (lib != tables.end())
            return lib->second;
        return add_table(key);
    }

private:
#ifdef _WIN64
    // Create a string with last error message
    std::string GetLastErrorStdStr() {
        DWORD error = GetLastError();
        if (error) {
            LPVOID lpMsgBuf;
            DWORD bufLen = FormatMessage(
                FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
                    FORMAT_MESSAGE_IGNORE_INSERTS,
                NULL, error, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPTSTR)&lpMsgBuf, 0, NULL);
            if (bufLen) {
                LPCSTR lpMsgStr = (LPCSTR)lpMsgBuf;
                std::string result(lpMsgStr, lpMsgStr + bufLen);

                LocalFree(lpMsgBuf);

                return result;
            }
        }
        return std::string();
    }
#endif

    function_table_t &add_table(oneapi::mkl::device key) {
        dlhandle handle;
        // check all available libraries for the key(device)
        for (const char *libname : libraries[domain_id][key]) {
            handle = dlhandle{ ::GET_LIB_HANDLE(libname) };
            if (handle)
                break;
        }
        if (!handle) {
            std::cerr << ERROR_MSG << '\n';
            throw mkl::backend_not_found();
        }
        auto t =
            reinterpret_cast<function_table_t *>(::GET_FUNC(handle.get(), table_names[domain_id]));

        if (!t) {
            std::cerr << ERROR_MSG << '\n';
            throw mkl::function_not_found();
        }
        if (t->version != SPEC_VERSION)
            throw mkl::specification_mismatch();

        handles[key] = std::move(handle);
        tables[key] = *t;
        return *t;
    }

    std::map<oneapi::mkl::device, function_table_t> tables;
    std::map<oneapi::mkl::device, dlhandle> handles;
};

} //namespace detail
} // namespace mkl
} // namespace oneapi

#endif //_LOADER_HPP_
