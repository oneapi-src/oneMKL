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

template <typename TypeIn, typename TypeOut>
void reference_forward_dft(std::vector<TypeIn> &in, std::vector<TypeOut> &out) {
    static_assert(is_complex<TypeOut>());

    double TWOPI = 2.0 * std::atan(1.0) * 4.0;

    std::complex<double> out_temp; /* Do the calculations using double */
    size_t N = out.size();
    for (int k = 0; k < N; k++) {
        out[k] = 0;
        out_temp = 0;
        for (int n = 0; n < N; n++) {
            if constexpr (is_complex<TypeIn>()) {
                out_temp += static_cast<std::complex<double>>(in[n]) *
                            std::complex<double>{ std::cos(n * k * TWOPI / N),
                                                  -std::sin(n * k * TWOPI / N) };
            }
            else {
                out_temp +=
                    std::complex<double>{ static_cast<double>(in[n]) * std::cos(n * k * TWOPI / N),
                                          static_cast<double>(-in[n]) *
                                              std::sin(n * k * TWOPI / N) };
            }
        }
        out[k] = static_cast<TypeOut>(out_temp);
    }
}

#endif //ONEMKL_REFERENCE_DFT_HPP
