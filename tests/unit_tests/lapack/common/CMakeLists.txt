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

# Build object from all test sources
set(LAPACK_COMMON_SOURCES "dependency_check.cpp" "global.cpp")

if(BUILD_SHARED_LIBS)
  target_sources(lapack_source_rt PRIVATE ${LAPACK_COMMON_SOURCES})
endif()

target_sources(lapack_source_ct PRIVATE ${LAPACK_COMMON_SOURCES})
