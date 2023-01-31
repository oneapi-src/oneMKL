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

#ifndef ONEMKL_REFERENCE_DFT_HPP
#define ONEMKL_REFERENCE_DFT_HPP

#include <cmath>
#include <complex>
#include <vector>

#include "test_common.hpp"

template <typename TypeIn, typename TypeOut>
void reference_forward_dft(const std::vector<TypeIn> &in, std::vector<TypeOut> &out) noexcept {
    using ref_t = long double; /* Do the calculations using long double */
    static_assert(is_complex<TypeOut>(), "Output type of DFT must be complex");

    const ref_t TWOPI = 2.0L * 3.141592653589793238462643383279502884197L;

    const size_t N = out.size();
    for (std::size_t k = 0; k < N; ++k) {
        std::complex<ref_t> out_temp = 0;
        const auto partial_expo = (static_cast<ref_t>(k) * TWOPI) / static_cast<ref_t>(N);
        for (std::size_t n = 0; n < N; ++n) {
            const auto expo = static_cast<ref_t>(n) * partial_expo;
            out_temp += static_cast<std::complex<ref_t>>(in[n]) *
                        std::complex<ref_t>{ std::cos(expo), -std::sin(expo) };
        }
        out[k] = static_cast<TypeOut>(out_temp);
    }
}

#endif //ONEMKL_REFERENCE_DFT_HPP
