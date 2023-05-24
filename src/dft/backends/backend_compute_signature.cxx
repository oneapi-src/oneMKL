/*******************************************************************************
* Copyright Codeplay Software Ltd.
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

/*
repetitive definitions from commit.cpp.

This file should be included for each backend, with <BACKEND> defined to match
the namespace of the backend's implementation.
*/

using scalar_type = typename dft::detail::commit_impl<prec, dom>::scalar_type;
using fwd_type = typename dft::detail::commit_impl<prec, dom>::fwd_type;
using bwd_type = typename dft::detail::commit_impl<prec, dom>::bwd_type;
using desc_type = typename dft::detail::descriptor<prec, dom>;

// forward inplace COMPLEX_COMPLEX
void forward_real(desc_type& desc, sycl::buffer<scalar_type, 1>& inout) override {
    oneapi::mkl::dft::BACKEND::compute_forward(desc, inout);
}
void forward_complex(desc_type& desc, sycl::buffer<bwd_type, 1>& inout) override {
    oneapi::mkl::dft::BACKEND::compute_forward(desc, inout);
}
sycl::event forward_real(desc_type& desc, scalar_type* inout,
                         const std::vector<sycl::event>& dependencies) override {
    return oneapi::mkl::dft::BACKEND::compute_forward(desc, inout, dependencies);
}
sycl::event forward_complex(desc_type& desc, bwd_type* inout,
                            const std::vector<sycl::event>& dependencies) override {
    return oneapi::mkl::dft::BACKEND::compute_forward(desc, inout, dependencies);
}

// forward inplace REAL_REAL
void forward_rr(desc_type& desc, sycl::buffer<scalar_type, 1>& inout_re,
                sycl::buffer<scalar_type, 1>& inout_im) override {
    oneapi::mkl::dft::BACKEND::compute_forward(desc, inout_re, inout_im);
}
sycl::event forward_rr(desc_type& desc, scalar_type* inout_re, scalar_type* inout_im,
                       const std::vector<sycl::event>& dependencies) override {
    return oneapi::mkl::dft::BACKEND::compute_forward(desc, inout_re, inout_im, dependencies);
}

// forward out-of-place COMPLEX_COMPLEX
void forward_oop_real(desc_type& desc, sycl::buffer<scalar_type, 1>& in,
                      sycl::buffer<scalar_type, 1>& out) override {
    oneapi::mkl::dft::BACKEND::compute_forward<desc_type, scalar_type, scalar_type>(desc, in, out);
}
void forward_oop_complex(desc_type& desc, sycl::buffer<bwd_type, 1>& in,
                         sycl::buffer<bwd_type, 1>& out) override {
    oneapi::mkl::dft::BACKEND::compute_forward<desc_type, bwd_type, bwd_type>(desc, in, out);
}
void forward_oop_mixed(desc_type& desc, sycl::buffer<fwd_type, 1>& in,
                       sycl::buffer<bwd_type, 1>& out) override {
    oneapi::mkl::dft::BACKEND::compute_forward<desc_type, fwd_type, bwd_type>(desc, in, out);
}
sycl::event forward_oop_real(desc_type& desc, scalar_type* in, scalar_type* out,
                             const std::vector<sycl::event>& dependencies) override {
    return oneapi::mkl::dft::BACKEND::compute_forward<desc_type, scalar_type, scalar_type>(
        desc, in, out, dependencies);
}
sycl::event forward_oop_complex(desc_type& desc, bwd_type* in, bwd_type* out,
                                const std::vector<sycl::event>& dependencies) override {
    return oneapi::mkl::dft::BACKEND::compute_forward<desc_type, bwd_type, bwd_type>(desc, in, out,
                                                                                     dependencies);
}
sycl::event forward_oop_mixed(desc_type& desc, fwd_type* in, bwd_type* out,
                              const std::vector<sycl::event>& dependencies) override {
    return oneapi::mkl::dft::BACKEND::compute_forward<desc_type, fwd_type, bwd_type>(desc, in, out,
                                                                                     dependencies);
}

// forward out-of-place REAL_REAL
void forward_oop_rr(desc_type& desc, sycl::buffer<scalar_type, 1>& in_re,
                    sycl::buffer<scalar_type, 1>& in_im, sycl::buffer<scalar_type, 1>& out_re,
                    sycl::buffer<scalar_type, 1>& out_im) override {
    oneapi::mkl::dft::BACKEND::compute_forward(desc, in_re, in_im, out_re, out_im);
}
sycl::event forward_oop_rr(desc_type& desc, scalar_type* in_re, scalar_type* in_im,
                           scalar_type* out_re, scalar_type* out_im,
                           const std::vector<sycl::event>& dependencies) override {
    return oneapi::mkl::dft::BACKEND::compute_forward(desc, in_re, in_im, out_re, out_im,
                                                      dependencies);
}

// backward inplace COMPLEX_COMPLEX
void backward_real(desc_type& desc, sycl::buffer<scalar_type, 1>& inout) override {
    oneapi::mkl::dft::BACKEND::compute_backward(desc, inout);
}
void backward_complex(desc_type& desc, sycl::buffer<bwd_type, 1>& inout) override {
    oneapi::mkl::dft::BACKEND::compute_backward(desc, inout);
}
sycl::event backward_real(desc_type& desc, scalar_type* inout,
                          const std::vector<sycl::event>& dependencies) override {
    return oneapi::mkl::dft::BACKEND::compute_backward(desc, inout, dependencies);
}
sycl::event backward_complex(desc_type& desc, bwd_type* inout,
                             const std::vector<sycl::event>& dependencies) override {
    return oneapi::mkl::dft::BACKEND::compute_backward(desc, inout, dependencies);
}

// backward inplace REAL_REAL
void backward_rr(desc_type& desc, sycl::buffer<scalar_type, 1>& inout_re,
                 sycl::buffer<scalar_type, 1>& inout_im) override {
    oneapi::mkl::dft::BACKEND::compute_backward(desc, inout_re, inout_im);
}
sycl::event backward_rr(desc_type& desc, scalar_type* inout_re, scalar_type* inout_im,
                        const std::vector<sycl::event>& dependencies) override {
    return oneapi::mkl::dft::BACKEND::compute_backward(desc, inout_re, inout_im, dependencies);
}

// backward out-of-place COMPLEX_COMPLEX
void backward_oop_real(desc_type& desc, sycl::buffer<scalar_type, 1>& in,
                       sycl::buffer<scalar_type, 1>& out) override {
    oneapi::mkl::dft::BACKEND::compute_backward<desc_type, scalar_type, scalar_type>(desc, in, out);
}
void backward_oop_complex(desc_type& desc, sycl::buffer<bwd_type, 1>& in,
                          sycl::buffer<bwd_type, 1>& out) override {
    oneapi::mkl::dft::BACKEND::compute_backward<desc_type, bwd_type, bwd_type>(desc, in, out);
}
void backward_oop_mixed(desc_type& desc, sycl::buffer<bwd_type, 1>& in,
                        sycl::buffer<fwd_type, 1>& out) override {
    oneapi::mkl::dft::BACKEND::compute_backward<desc_type, bwd_type, fwd_type>(desc, in, out);
}
sycl::event backward_oop_real(desc_type& desc, scalar_type* in, scalar_type* out,
                              const std::vector<sycl::event>& dependencies) override {
    return oneapi::mkl::dft::BACKEND::compute_backward<desc_type, scalar_type, scalar_type>(
        desc, in, out, dependencies);
}
sycl::event backward_oop_complex(desc_type& desc, bwd_type* in, bwd_type* out,
                                 const std::vector<sycl::event>& dependencies) override {
    return oneapi::mkl::dft::BACKEND::compute_backward<desc_type, bwd_type, bwd_type>(desc, in, out,
                                                                                      dependencies);
}
sycl::event backward_oop_mixed(desc_type& desc, bwd_type* in, fwd_type* out,
                               const std::vector<sycl::event>& dependencies) override {
    return oneapi::mkl::dft::BACKEND::compute_backward<desc_type, bwd_type, fwd_type>(desc, in, out,
                                                                                      dependencies);
}

// backward out-of-place REAL_REAL
void backward_oop_rr(desc_type& desc, sycl::buffer<scalar_type, 1>& in_re,
                     sycl::buffer<scalar_type, 1>& in_im, sycl::buffer<scalar_type, 1>& out_re,
                     sycl::buffer<scalar_type, 1>& out_im) override {
    oneapi::mkl::dft::BACKEND::compute_backward(desc, in_re, in_im, out_re, out_im);
}
sycl::event backward_oop_rr(desc_type& desc, scalar_type* in_re, scalar_type* in_im,
                            scalar_type* out_re, scalar_type* out_im,
                            const std::vector<sycl::event>& dependencies) override {
    return oneapi::mkl::dft::BACKEND::compute_backward(desc, in_re, in_im, out_re, out_im,
                                                       dependencies);
}
