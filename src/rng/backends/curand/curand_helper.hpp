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

/**
 * @file mkl_rng_curand.cpp : contains the implementation of all the routines
 * for CUDA backend
 */
#ifndef _MKL_RNG_CURAND_HELPER_HPP_
#define _MKL_RNG_CURAND_HELPER_HPP_
#include <curand.h>
#include <cuda.h>
#include <complex>
#include "oneapi/mkl/types.hpp"

namespace oneapi {
namespace mkl {
namespace rng {
namespace curand {

class curand_error : virtual public std::runtime_error {
protected:
    inline const char *curand_error_map(curandStatus_t error) {
        switch (error) {
            case CURAND_STATUS_SUCCESS: return "CURAND_STATUS_SUCCESS";

            case CURAND_STATUS_VERSION_MISMATCH: return "CURAND_STATUS_VERSION_MISMATCH";

            case CURAND_STATUS_NOT_INITIALIZED: return "CURAND_STATUS_NOT_INITIALIZED";

            case CURAND_STATUS_ALLOCATION_FAILED: return "CURAND_STATUS_ALLOCATION_FAILED";

            case CURAND_STATUS_TYPE_ERROR: return "CURAND_STATUS_TYPE_ERROR";

            case CURAND_STATUS_OUT_OF_RANGE: return "CURAND_STATUS_OUT_OF_RANGE";

            case CURAND_STATUS_LENGTH_NOT_MULTIPLE: return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";

            case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
                return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";

            case CURAND_STATUS_LAUNCH_FAILURE: return "CURAND_STATUS_LAUNCH_FAILURE";

            case CURAND_STATUS_PREEXISTING_FAILURE: return "CURAND_STATUS_PREEXISTING_FAILURE";

            case CURAND_STATUS_INITIALIZATION_FAILED: return "CURAND_STATUS_INITIALIZATION_FAILED";

            case CURAND_STATUS_ARCH_MISMATCH: return "CURAND_STATUS_ARCH_MISMATCH";

            case CURAND_STATUS_INTERNAL_ERROR: return "CURAND_STATUS_INTERNAL_ERROR";

            default: return "<unknown>";
        }
    }

    int error_number; ///< Error number
public:
    /** Constructor (C++ STL string, curandStatus_t).
   *  @param msg The error message
   *  @param err_num error number
   */
    explicit curand_error(std::string message, curandStatus_t result)
            : std::runtime_error((message + std::string(curand_error_map(result)))) {
        error_number = static_cast<int>(result);
    }

    /** Destructor.
   *  Virtual to allow for subclassing.
   */
    virtual ~curand_error() throw() {}

    /** Returns error number.
   *  @return #error_number
   */
    virtual int getErrorNumber() const throw() {
        return error_number;
    }
};

class cuda_error : virtual public std::runtime_error {
protected:
    inline const char *cuda_error_map(CUresult result) {
        switch (result) {
            case CUDA_SUCCESS: return "CUDA_SUCCESS";

            case CUDA_ERROR_NOT_PERMITTED: return "CUDA_ERROR_NOT_PERMITTED";

            case CUDA_ERROR_INVALID_CONTEXT: return "CUDA_ERROR_INVALID_CONTEXT";

            case CUDA_ERROR_INVALID_DEVICE: return "CUDA_ERROR_INVALID_DEVICE";

            case CUDA_ERROR_INVALID_VALUE: return "CUDA_ERROR_INVALID_VALUE";

            case CUDA_ERROR_OUT_OF_MEMORY: return "CUDA_ERROR_OUT_OF_MEMORY";

            case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES: return "CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES";

            default: return "<unknown>";
        }
    }
    int error_number; ///< error number
public:
    /** Constructor (C++ STL string, CUresult).
   *  @param msg The error message
   *  @param err_num Error number
   */
    explicit cuda_error(std::string message, CUresult result)
            : std::runtime_error((message + std::string(cuda_error_map(result)))) {
        error_number = static_cast<int>(result);
    }

    /** Destructor.
   *  Virtual to allow for subclassing.
   */
    virtual ~cuda_error() throw() {}

    /** Returns error number.
   *  @return #error_number
   */
    virtual int getErrorNumber() const throw() {
        return error_number;
    }
};

#define CUDA_ERROR_FUNC(name, err, ...)                                 \
    err = name(__VA_ARGS__);                                            \
    if (err != CUDA_SUCCESS) {                                          \
        throw cuda_error(std::string(#name) + std::string(" : "), err); \
    }

#define CURAND_CALL(func, status, ...)                                       \
    status = func(__VA_ARGS__);                                              \
    if (status != CURAND_STATUS_SUCCESS) {                                   \
        throw curand_error(std::string(#func) + std::string(" : "), status); \
    }

// inline curandGeneratorType_t get_curand_generator() {
//     return CURAND_RNG_TEST;
// }

// inline curandRngType_t get_curand_rng() {
//     CURAND_RNG_TEST = 0,
//     CURAND_RNG_PSEUDO_DEFAULT = 100, ///< Default pseudorandom generator
//         CURAND_RNG_PSEUDO_XORWOW = 101, ///< XORWOW pseudorandom generator
//         CURAND_RNG_PSEUDO_MRG32K3A = 121, ///< MRG32k3a pseudorandom generator
//         CURAND_RNG_PSEUDO_MTGP32 = 141, ///< Mersenne Twister MTGP32 pseudorandom generator
//         CURAND_RNG_PSEUDO_MT19937 = 142, ///< Mersenne Twister MT19937 pseudorandom generator
//         CURAND_RNG_PSEUDO_PHILOX4_32_10 = 161, ///< PHILOX-4x32-10 pseudorandom generator
//         CURAND_RNG_QUASI_DEFAULT = 200, ///< Default quasirandom generator
//         CURAND_RNG_QUASI_SOBOL32 = 201, ///< Sobol32 quasirandom generator
//         CURAND_RNG_QUASI_SCRAMBLED_SOBOL32 = 202, ///< Scrambled Sobol32 quasirandom generator
//         CURAND_RNG_QUASI_SOBOL64 = 203, ///< Sobol64 quasirandom generator
//         CURAND_RNG_QUASI_SCRAMBLED_SOBOL64 = 204 ///< Scrambled Sobol64 quasirandom generator
// }

} // namespace curand
} // namespace rng
} // namespace mkl
} // namespace oneapi

#endif // _MKL_RNG_CURAND_HELPER_HPP_
