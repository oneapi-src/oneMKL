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

#ifndef _ONEMKL_EXCEPTIONS_HPP_
#define _ONEMKL_EXCEPTIONS_HPP_

#include "onemkl/detail/backends.hpp"
#include "onemkl/types.hpp"

namespace onemkl {

class InvalidArgumentsException : virtual public std::exception {
private:
    std::string error_message = "";

public:
    InvalidArgumentsException(const std::string& msg)
            : error_message(std::string("oneMKL InvalidArgumentsException: \n") + msg + "\n") {}

    const char* what() const noexcept override {
        return error_message.c_str();
    };
};

class MemoryAllocationException : virtual public std::exception {
private:
    std::string error_message = "";

public:
    MemoryAllocationException(const std::string& msg)
            : error_message(std::string("oneMKL MemoryAllocationException: \n") + msg + "\n") {}

    const char* what() const noexcept override {
        return error_message.c_str();
    };
};

class UnsupportedBackendException : virtual public std::exception {
private:
    std::string error_message = "";

public:
    UnsupportedBackendException(cl::sycl::queue& queue, const std::string& msg)
            : error_message(
                  std::string("oneMKL UnsupportedBackendException: \n") +
                  std::string("  There is currently no onemkl::backend available for the \n") +
                  std::string("  provided queue, device and sycl::backend. \n")) {}

    const char* what() const noexcept override {
        return error_message.c_str();
    };
};

class BackendNotAvailableForApiException : virtual public std::exception {
private:
    std::string error_message = "";

public:
    BackendNotAvailableForApiException(cl::sycl::queue& queue, onemkl::backend& backend,
                                       const std::string& api_description)
            : error_message(std::string("oneMKL BackendNotAvailableForApiException: \n") +
                            std::string("  The onemkl::backend = ") + onemkl::backend_map[backend] +
                            std::string("\n") + std::string("  is not available for ") +
                            api_description + std::string("\n")) {}

    const char* what() const noexcept override {
        return error_message.c_str();
    };
};

} //namespace onemkl

#endif //_ONEMKL_EXCEPTIONS_HPP_
