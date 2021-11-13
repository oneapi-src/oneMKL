# Copyright (c) 2019-2020 Intel Corporation.
# SPDX-License-Identifier: BSD-3-Clause

# Get Base image from Docker Hub Intel repo
ARG base_image="intel/oneapi:os-tools-ubuntu18.04"
FROM "$base_image"

ARG DEBIAN_FRONTEND=noninteractive
ARG APT_KEY_DONT_WARN_ON_DANGEROUS_USAGE=1

# add apt repo public key
ARG url=https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2023.PUB
ADD $url /
RUN file=$(basename "$url") && \
    apt-key add "$file" && \
    rm "$file"

# configure the repository
ARG apt_repo=https://apt.repos.intel.com/oneapi
RUN echo "deb $apt_repo all main" > /etc/apt/sources.list.d/oneAPI.list

# install Intel(R) oneAPI essentials
RUN apt-get update -y && \
    apt-get install -y -o=Dpkg::Use-Pty=0 \
    wget python3-pip git gfortran \
        intel-oneapi-common-vars \
        intel-oneapi-common-licensing \
        intel-oneapi-mkl-devel \
        intel-oneapi-tbb-devel \
        intel-oneapi-libdpstd-devel \
        intel-oneapi-compiler-dpcpp-cpp \
--

# install Intel(R) Graphics Compute Runtime for OpenCL(TM)
# https://github.com/intel/compute-runtime/releases/
RUN mkdir neo && \
    cd neo && \
    wget --no-verbose \
        https://github.com/intel/compute-runtime/releases/download/21.12.19358/intel-gmmlib_20.3.2_amd64.deb \
        https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.6712/intel-igc-core_1.0.6712_amd64.deb \
        https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.6712/intel-igc-opencl_1.0.6712_amd64.deb \
        https://github.com/intel/compute-runtime/releases/download/21.12.19358/intel-opencl_21.12.19358_amd64.deb \
        https://github.com/intel/compute-runtime/releases/download/21.12.19358/intel-ocloc_21.12.19358_amd64.deb \
        https://github.com/intel/compute-runtime/releases/download/21.12.19358/intel-level-zero-gpu_1.0.19358_amd64.deb \
        https://github.com/intel/compute-runtime/releases/download/21.12.19358/ww12.sum \
    && \
    sha256sum -c ww12.sum && \
    dpkg -i *.deb && \
    cd .. && rm -r neo && \
    apt-get purge -y wget

# Install NVIDIA CUDA SDK 10.1
#RUN apt-get install -y -o=Dpkg::Use-Pty=0 wget && \
#wget --no-verbose \
#    https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.1.105-1_amd64.deb
#RUN dpkg -i cuda-repo-ubuntu1804_10.1.105-1_amd64.deb
#RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
#RUN apt-get update
#RUN apt-get install -y cuda

# Install Conan Package Manager
RUN pip3 install conan==1.33 && conan --version

# entrypoint
#SHELL ["/bin/bash", "-c"]
RUN mkdir -p /usr/onemkl/BUILD
COPY ./ /usr/onemkl/

# Setup DPC++ compiler prefix for Conan
RUN sed -i 's#<path to dpc++ compiler root>#/opt/intel/oneapi/compiler/latest/linux/#g' /usr/onemkl/conan/profiles/inteldpcpp_lnx


#RUN cd /usr/onemkl/BUILD && \
#    /bin/bash -c "source /opt/intel/oneapi/setvars.sh" && \
#    conan config install ../conan && \
#    /bin/bash -c "conan install .. -pr inteldpcpp_lnx --build missing"

#COPY entrypoint.sh /
#RUN chmod a+x entrypoint.sh
#ENTRYPOINT ["/entrypoint.sh"]
#CMD ["/bin/bash"]
