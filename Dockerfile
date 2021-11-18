# Copyright (c) 2019-2020 Intel Corporation.
# SPDX-License-Identifier: BSD-3-Clause

# Get Base image from Docker Hub Intel repo
ARG base_image="intel/oneapi-basekit"
FROM "$base_image"

ARG DEBIAN_FRONTEND=noninteractive
ARG APT_KEY_DONT_WARN_ON_DANGEROUS_USAGE=1

# Install build essentials
RUN apt-get update -y && \
    apt-get install -y -o=Dpkg::Use-Pty=0 \
    python3-pip
    # gfortran

# Install NVIDIA CUDA SDK 10.1
#RUN apt-get install -y -o=Dpkg::Use-Pty=0 wget && \
#wget --no-verbose \
#    https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.1.105-1_amd64.deb
#RUN dpkg -i cuda-repo-ubuntu1804_10.1.105-1_amd64.deb
#RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
#RUN apt-get update
#RUN apt-get install -y cuda

# Install Conan Package Manager
RUN pip3 install conan && conan --version

# Setup Working Directory
COPY . /home/oneMKL
WORKDIR /home/oneMKL

# Setup DPC++ compiler and Conan build-profiles
RUN sed -i 's#<path to dpc++ compiler root>#/opt/intel/oneapi/compiler/latest/linux/#g' conan/profiles/inteldpcpp_lnx && cat conan/profiles/inteldpcpp_lnx
RUN mkdir -p BUILD && cd BUILD && conan config install ../conan

# Can't get Entrypoint to work with Jenkins
# Uncomment for local use or development
#ENTRYPOINT ["/home/oneMKL/docker/entrypoint.sh"]
