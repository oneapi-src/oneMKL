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

#include <iostream>
#include <vector>

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include "test_helper.hpp"
#include "test_common.hpp"
#include "parseval_check.hpp"
#include <gtest/gtest.h>

extern std::vector<sycl::device*> devices;

class WorkspaceExternalTests : public ::testing::TestWithParam<sycl::device*> {};

template <oneapi::mkl::dft::precision prec, oneapi::mkl::dft::domain dom>
int test_workspace_external_usm_impl(std::size_t dftSize, sycl::device* dev) {
    using namespace oneapi::mkl::dft;
    using scalar_t = std::conditional_t<prec == precision::DOUBLE, double, float>;
    using forward_t = std::conditional_t<dom == domain::COMPLEX, std::complex<scalar_t>, scalar_t>;
    using backward_t = std::complex<scalar_t>;

    sycl::queue sycl_queue(*dev);
    if (prec == precision::DOUBLE && !sycl_queue.get_device().has(sycl::aspect::fp64)) {
        return test_skipped;
    }
    descriptor<prec, dom> desc(static_cast<std::int64_t>(dftSize));

    desc.set_value(config_param::WORKSPACE_PLACEMENT, config_value::WORKSPACE_EXTERNAL);
    desc.set_value(config_param::PLACEMENT, config_value::NOT_INPLACE);
    commit_descriptor(desc, sycl_queue);
    std::int64_t workspaceBytes = -1;
    desc.get_value(config_param::WORKSPACE_EXTERNAL_BYTES, &workspaceBytes);
    if (workspaceBytes < 0) {
        return ::testing::Test::HasFailure();
    }
    scalar_t* workspace = sycl::malloc_device<scalar_t>(
        static_cast<std::size_t>(workspaceBytes) / sizeof(scalar_t), sycl_queue);
    desc.set_workspace(workspace);
    // Generate data
    std::vector<forward_t> hostFwd(static_cast<std::size_t>(dftSize));
    std::size_t bwdSize = dom == domain::COMPLEX ? dftSize : dftSize / 2 + 1;
    std::vector<backward_t> hostBwd(bwdSize);
    rand_vector(hostFwd, dftSize);

    // Allocate enough memory that we don't have to worry about the domain.
    forward_t* deviceFwd = sycl::malloc_device<forward_t>(dftSize, sycl_queue);
    backward_t* deviceBwd = sycl::malloc_device<backward_t>(bwdSize, sycl_queue);
    sycl_queue.copy(hostFwd.data(), deviceFwd, dftSize);
    sycl_queue.wait_and_throw();

    compute_forward<decltype(desc), forward_t, backward_t>(desc, deviceFwd, deviceBwd);
    sycl_queue.wait_and_throw();

    sycl_queue.copy(deviceBwd, hostBwd.data(), bwdSize);
    sycl_queue.wait_and_throw();

    // To see external workspaces, larger sizes of DFT may be needed. Using the reference DFT with larger sizes is slow,
    // so use Parseval's theorum as a sanity check instead.
    bool sanityCheckPasses = parseval_check(dftSize, hostFwd.data(), hostBwd.data());

    if (sanityCheckPasses) {
        sycl_queue.copy(hostFwd.data(), deviceFwd, dftSize);
        sycl_queue.wait_and_throw();
        compute_backward<decltype(desc), backward_t, forward_t>(desc, deviceBwd, deviceFwd);
        sycl_queue.wait_and_throw();
        sycl_queue.copy(deviceFwd, hostFwd.data(), dftSize);
        sycl_queue.wait_and_throw();
        forward_t rescale =
            static_cast<forward_t>(1) / static_cast<forward_t>(static_cast<scalar_t>(dftSize));
        sanityCheckPasses = parseval_check(dftSize, hostFwd.data(), hostBwd.data(), rescale);
    }

    sycl::free(deviceFwd, sycl_queue);
    sycl::free(deviceBwd, sycl_queue);
    sycl::free(workspace, sycl_queue);
    return sanityCheckPasses ? !::testing::Test::HasFailure() : ::testing::Test::HasFailure();
    ;
}

template <oneapi::mkl::dft::precision prec, oneapi::mkl::dft::domain dom>
int test_workspace_external_buffer_impl(std::size_t dftSize, sycl::device* dev) {
    using namespace oneapi::mkl::dft;
    using scalar_t = std::conditional_t<prec == precision::DOUBLE, double, float>;
    using forward_t = std::conditional_t<dom == domain::COMPLEX, std::complex<scalar_t>, scalar_t>;
    using backward_t = std::complex<scalar_t>;

    sycl::queue sycl_queue(*dev);
    if (prec == precision::DOUBLE && !sycl_queue.get_device().has(sycl::aspect::fp64)) {
        return test_skipped;
    }
    descriptor<prec, dom> desc(static_cast<std::int64_t>(dftSize));

    desc.set_value(config_param::WORKSPACE_PLACEMENT, config_value::WORKSPACE_EXTERNAL);
    desc.set_value(config_param::PLACEMENT, config_value::NOT_INPLACE);
    commit_descriptor(desc, sycl_queue);
    std::int64_t workspaceBytes = -1;
    desc.get_value(config_param::WORKSPACE_EXTERNAL_BYTES, &workspaceBytes);
    if (workspaceBytes < 0) {
        return ::testing::Test::HasFailure();
    }
    sycl::buffer<scalar_t> workspace(static_cast<std::size_t>(workspaceBytes) / sizeof(scalar_t));
    desc.set_workspace(workspace);
    // Generate data
    std::vector<forward_t> hostFwd(static_cast<std::size_t>(dftSize));
    std::size_t bwdSize = dom == domain::COMPLEX ? dftSize : dftSize / 2 + 1; // TODO: Check this!
    std::vector<backward_t> hostBwd(bwdSize);
    rand_vector(hostFwd, dftSize);
    auto hostFwdCpy = hostFwd; // Some backends modify the input data (rocFFT).

    {
        sycl::buffer<forward_t> bufFwd(hostFwd);
        sycl::buffer<backward_t> bufBwd(hostBwd);
        compute_forward<decltype(desc), forward_t, backward_t>(desc, bufFwd, bufBwd);
    }

    // To see external workspaces, larger sizes of DFT may be needed. Using the reference DFT with larger sizes is slow,
    // so use Parseval's theorum as a sanity check instead.
    bool sanityCheckPasses = parseval_check(dftSize, hostFwdCpy.data(), hostBwd.data());

    if (sanityCheckPasses) {
        auto hostBwdCpy = hostBwd;
        {
            sycl::buffer<forward_t> bufFwd(hostFwd);
            sycl::buffer<backward_t> bufBwd(hostBwd);
            compute_backward<decltype(desc), backward_t, forward_t>(desc, bufBwd, bufFwd);
            sycl_queue.wait_and_throw();
        }
        forward_t rescale =
            static_cast<forward_t>(1) / static_cast<forward_t>(static_cast<scalar_t>(dftSize));
        sanityCheckPasses = parseval_check(dftSize, hostFwd.data(), hostBwdCpy.data(), rescale);
    }

    return sanityCheckPasses ? !::testing::Test::HasFailure() : ::testing::Test::HasFailure();
    ;
}

template <oneapi::mkl::dft::precision prec, oneapi::mkl::dft::domain dom>
void test_workspace_external_usm(sycl::device* dev) {
    EXPECT_TRUEORSKIP((test_workspace_external_usm_impl<prec, dom>(2, dev)));
    EXPECT_TRUEORSKIP((test_workspace_external_usm_impl<prec, dom>(1024 * 3 * 5 * 7 * 16, dev)));
}

template <oneapi::mkl::dft::precision prec, oneapi::mkl::dft::domain dom>
void test_workspace_external_buffer(sycl::device* dev) {
    EXPECT_TRUEORSKIP((test_workspace_external_buffer_impl<prec, dom>(2, dev)));
    EXPECT_TRUEORSKIP((test_workspace_external_buffer_impl<prec, dom>(1024 * 3 * 5 * 7 * 16, dev)));
}

TEST_P(WorkspaceExternalTests, TestWorkspaceExternalSingleUsm) {
    using precision = oneapi::mkl::dft::precision;
    using domain = oneapi::mkl::dft::domain;
    test_workspace_external_usm<precision::SINGLE, domain::REAL>(GetParam());
    test_workspace_external_usm<precision::SINGLE, domain::COMPLEX>(GetParam());
}

TEST_P(WorkspaceExternalTests, TestWorkspaceExternalDoubleUsm) {
    using precision = oneapi::mkl::dft::precision;
    using domain = oneapi::mkl::dft::domain;
    test_workspace_external_usm<precision::DOUBLE, domain::REAL>(GetParam());
    test_workspace_external_usm<precision::DOUBLE, domain::COMPLEX>(GetParam());
}

TEST_P(WorkspaceExternalTests, TestWorkspaceExternalSingleBuffer) {
    using precision = oneapi::mkl::dft::precision;
    using domain = oneapi::mkl::dft::domain;
    test_workspace_external_buffer<precision::SINGLE, domain::REAL>(GetParam());
    test_workspace_external_buffer<precision::SINGLE, domain::COMPLEX>(GetParam());
}

TEST_P(WorkspaceExternalTests, TestWorkspaceExternalDoubleBuffer) {
    using precision = oneapi::mkl::dft::precision;
    using domain = oneapi::mkl::dft::domain;
    test_workspace_external_buffer<precision::DOUBLE, domain::REAL>(GetParam());
    test_workspace_external_buffer<precision::DOUBLE, domain::COMPLEX>(GetParam());
}

/// A test where set_workspace is called when an external workspace is not set.
TEST_P(WorkspaceExternalTests, SetWorkspaceOnWorkspaceAutomatic) {
    using namespace oneapi::mkl::dft;
    sycl::queue sycl_queue(*GetParam());
    const int dftLen = 1024 * 3 * 5 * 7 * 16; // A size likely to require an external workspace.
    float* fftDataUsm = sycl::malloc_device<float>(dftLen * 2, sycl_queue);
    sycl::buffer<float> fftDataBuf(dftLen * 2);
    descriptor<precision::SINGLE, domain::COMPLEX> descUsm(dftLen), descBuf(dftLen);
    // WORKSPACE_EXTERNAL is NOT set.
    commit_descriptor(descUsm, sycl_queue);
    commit_descriptor(descBuf, sycl_queue);
    std::int64_t workspaceBytes = 0;
    descUsm.get_value(config_param::WORKSPACE_EXTERNAL_BYTES, &workspaceBytes);

    // No workspace set yet: all of the following should work.
    compute_forward(descUsm, fftDataUsm);
    compute_forward(descBuf, fftDataBuf);
    compute_backward(descUsm, fftDataUsm);
    compute_backward(descBuf, fftDataBuf);
    compute_forward(descUsm, fftDataBuf);
    compute_forward(descBuf, fftDataUsm);
    compute_backward(descUsm, fftDataBuf);
    compute_backward(descBuf, fftDataUsm);
    sycl_queue.wait_and_throw();

    // Set workspace
    float* usmWorkspace = sycl::malloc_device<float>(
        static_cast<std::size_t>(workspaceBytes) / sizeof(float), sycl_queue);
    sycl::buffer<float> bufferWorkspace(static_cast<std::size_t>(workspaceBytes) / sizeof(float));
    descUsm.set_workspace(usmWorkspace);
    descBuf.set_workspace(bufferWorkspace);

    // Should work:
    compute_forward(descUsm, fftDataUsm);
    sycl_queue.wait_and_throw();
    compute_forward(descBuf, fftDataBuf);
    sycl_queue.wait_and_throw();
    compute_backward(descUsm, fftDataUsm);
    sycl_queue.wait_and_throw();
    compute_backward(descBuf, fftDataBuf);
    sycl_queue.wait_and_throw();

    // Should not work:
    EXPECT_THROW(compute_forward(descUsm, fftDataBuf), oneapi::mkl::invalid_argument);
    EXPECT_THROW(compute_forward(descBuf, fftDataUsm), oneapi::mkl::invalid_argument);
    EXPECT_THROW(compute_backward(descUsm, fftDataBuf), oneapi::mkl::invalid_argument);
    EXPECT_THROW(compute_backward(descBuf, fftDataUsm), oneapi::mkl::invalid_argument);
    sycl_queue.wait_and_throw();

    // Free any allocations:
    sycl::free(usmWorkspace, sycl_queue);
    sycl::free(fftDataUsm, sycl_queue);
}

/// Test that the implementation throws as expected.
TEST_P(WorkspaceExternalTests, ThrowOnBadCalls) {
    using namespace oneapi::mkl::dft;
    sycl::queue sycl_queue(*GetParam());
    const int dftLen = 1024 * 3 * 5 * 7 * 16; // A size likely to require an external workspace.
    float* fftDataUsm = sycl::malloc_device<float>(dftLen * 2, sycl_queue);
    sycl::buffer<float> fftDataBuf(dftLen * 2);
    descriptor<precision::SINGLE, domain::COMPLEX> descUsm(dftLen), descBuf(dftLen);
    descUsm.set_value(config_param::WORKSPACE_PLACEMENT, config_value::WORKSPACE_EXTERNAL);
    descBuf.set_value(config_param::WORKSPACE_PLACEMENT, config_value::WORKSPACE_EXTERNAL);

    // We expect the following to throw because the decriptor has not been committed.
    std::int64_t workspaceBytes = -10;
    float* usmWorkspace = nullptr;
    EXPECT_THROW(descUsm.get_value(config_param::WORKSPACE_EXTERNAL_BYTES, &workspaceBytes),
                 oneapi::mkl::invalid_argument);
    EXPECT_THROW(descUsm.set_workspace(usmWorkspace), oneapi::mkl::uninitialized);
    commit_descriptor(descUsm, sycl_queue);
    commit_descriptor(descBuf, sycl_queue);

    descUsm.get_value(config_param::WORKSPACE_EXTERNAL_BYTES, &workspaceBytes);
    EXPECT_GE(workspaceBytes, 0);

    // We haven't set a workspace, so the following should fail;
    EXPECT_THROW(compute_forward(descUsm, fftDataUsm), oneapi::mkl::invalid_argument);
    sycl_queue.wait_and_throw();
    EXPECT_THROW(compute_forward(descUsm, fftDataBuf), oneapi::mkl::invalid_argument);
    sycl_queue.wait_and_throw();

    if (workspaceBytes > 0) {
        EXPECT_THROW(descUsm.set_workspace(nullptr), oneapi::mkl::invalid_argument);
        sycl::buffer<float> undersizeWorkspace(
            static_cast<std::size_t>(workspaceBytes) / sizeof(float) - 1);
        EXPECT_THROW(descBuf.set_workspace(undersizeWorkspace), oneapi::mkl::invalid_argument);
    }

    usmWorkspace = sycl::malloc_device<float>(
        static_cast<std::size_t>(workspaceBytes) / sizeof(float), sycl_queue);
    sycl::buffer<float> bufferWorkspace(static_cast<std::size_t>(workspaceBytes) / sizeof(float));

    descUsm.set_workspace(usmWorkspace);
    descBuf.set_workspace(bufferWorkspace);

    // Should work:
    compute_forward(descUsm, fftDataUsm);
    sycl_queue.wait_and_throw();
    compute_forward(descBuf, fftDataBuf);
    sycl_queue.wait_and_throw();
    compute_backward(descUsm, fftDataUsm);
    sycl_queue.wait_and_throw();
    compute_backward(descBuf, fftDataBuf);
    sycl_queue.wait_and_throw();

    // Should not work:
    EXPECT_THROW(compute_forward(descUsm, fftDataBuf), oneapi::mkl::invalid_argument);
    EXPECT_THROW(compute_forward(descBuf, fftDataUsm), oneapi::mkl::invalid_argument);
    EXPECT_THROW(compute_backward(descUsm, fftDataBuf), oneapi::mkl::invalid_argument);
    EXPECT_THROW(compute_backward(descBuf, fftDataUsm), oneapi::mkl::invalid_argument);
    sycl_queue.wait_and_throw();

    // Free any allocations:
    sycl::free(usmWorkspace, sycl_queue);
    sycl::free(fftDataUsm, sycl_queue);
}

INSTANTIATE_TEST_SUITE_P(WorkspaceExternalTestSuite, WorkspaceExternalTests,
                         testing::ValuesIn(devices), ::DeviceNamePrint());
