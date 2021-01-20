#===============================================================================
# Copyright 2020-2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.
#
#
# SPDX-License-Identifier: Apache-2.0
#===============================================================================

from conans import ConanFile, CMake, tools
from packaging.version import parse
from six import StringIO

class oneMKLConan(ConanFile):
    name = "oneMKL"
    version = "0.1.0-beta"
    url = "https://github.com/oneapi-src/oneMKL"
    description = "oneMKL interfaces is an open-source implementation of oneMKL Data Parallel C++ (DPC++) interfaces according to oneMKL specification that can work with multiple devices (backends) using device specific libraries underneath."
    license = "Apache License Version 2.0"
    legal_notice = "\
LEGAL NOTICE: By downloading and using this container or script as applicable \n \
(the “Software Package”), and any included software or software made available \n \
for download, you agree to the terms and conditions of the software license \n \
agreements for the Software Package, which may also include notices, \n \
disclaimers or license terms for third party software (the “Agreements”) \n \
located at https://github.com/conan-io/conan/blob/develop/LICENSE.md, the \n \
THIRD-PARTY-PROGRAMS file and in the README.md file included with the Software Package. \n \
    "

    # Dependencies
    netlib_version = "3.7.1"
    sphinx_version = "2.4.4"

    settings = "os", "compiler", "build_type", "arch"
    options = {
        # Build style
        "build_shared_libs": [True, False],
        "target_domains"   : "ANY",

        # Backends
        "enable_mklcpu_backend"   : [True, False],
        "enable_mklgpu_backend"   : [True, False],

        # Threading for mklcpu_backend
        "enable_mklcpu_thread_tbb": [True, False],

        # Testing
        "build_functional_tests"  : [True, False],

        # Documentation
        "build_doc"        : [True, False]
    }
    default_options = {
        "build_shared_libs"       : True,
        "target_domains"          : None,

        "enable_mklcpu_backend"   : True,
        "enable_mklgpu_backend"   : True,

        "enable_mklcpu_thread_tbb": True,

        "build_functional_tests" : True,

        "build_doc"        : False,

        # External package options
        "lapack:shared": True
    }
    generators = "cmake"
    no_copy_source = True
    exports_sources = "cmake/*", "include/*", "tests/*", "CMakeLists.txt"


    def system_requirements(self):
        self.output.info(f"\n {self.legal_notice}")
        self.global_system_requirements = True
        installer = tools.SystemPackageTool()
        if self.options.enable_mklcpu_backend or self.options.enable_mklgpu_backend:
            installer.add_repository("\"deb https://apt.repos.intel.com/oneapi all main\"")
            installer.install("intel-oneapi-mkl-devel")  # User must apt-key add GPG key before they can download oneMKL
            if self.options.enable_mklcpu_thread_tbb:
                installer.install("intel-oneapi-tbb-devel")  # For libtbb.so used during link-time


    def get_python_exe(self):
        # Find supported Python binary
        python_exe = "python3"
        try:
            self.run(f"{python_exe} --version", output=False)
        except:
            python_exe = "python"
            my_buffer = StringIO()
            self.run(f"{python_exe} --version", output=my_buffer)
            ver_found = parse( my_buffer.getvalue().replace('Python ', '') )
            if ver_found < parse('3.6.0'):
                self.output.error(f"Python 3.6.0 or higher required. Found {ver_found}")
                return
        return python_exe


    def build_requirements(self):
        if self.options.build_functional_tests:
            self.build_requires(f"lapack/{self.netlib_version}@conan/stable")
        # For Sphinx only
        if self.options.build_doc:
            # Use pip to install Sphinx as a user package
            self.run(f"{self.get_python_exe()} -m pip install sphinx=={self.sphinx_version}")


    def _cmake(self):
        cmake = CMake(self)
        return cmake


    def build(self):
        cmake = self._cmake()
        cmake.definitions.update({
            # Options
            "BUILD_SHARED_LIBS"        : self.options.build_shared_libs,
            "ENABLE_MKLCPU_BACKEND"    : self.options.enable_mklcpu_backend,
            "ENABLE_MKLGPU_BACKEND"    : self.options.enable_mklgpu_backend,
            "ENABLE_MKLCPU_THREAD_TBB" : self.options.enable_mklcpu_thread_tbb,
            "BUILD_FUNCTIONAL_TESTS"   : self.options.build_functional_tests,
            "BUILD_DOC"                : self.options.build_doc,

            # Paramaters
            # Conan does not officially support oneAPI DPC++ Compiler, hence disable the compiler_id check
            "CONAN_DISABLE_CHECK_COMPILER" : True,
            "MKL_ROOT"                 : "/opt/intel/inteloneapi/mkl/latest",
        })
        # Pass target_domains definition transparently to CMake, since CMakeLists.txt contains checks and defaults.
        if self.options.target_domains != None:
            cmake.definitions["TARGET_DOMAINS"] = self.options.target_domains
        cmake.configure()
        cmake.build()
        if tools.get_env("CONAN_RUN_TESTS", default=True) and self.options.build_functional_tests:
            cmake.test()


    def package(self):
        cmake = self._cmake()
        cmake.install()


    def package_info(self):
        self.cpp_info.libs = ["onemkl"]
