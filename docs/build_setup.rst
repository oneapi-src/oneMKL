.. _build_setup:

Build Setup
===========

#. 
   Install Intel(R) oneAPI DPC++ Compiler (select variant as per requirement).

#. 
   Clone this project to ``<path to onemkl>``\ , where ``<path to onemkl>``
   is the root directory of this repository.

#. 
   You can `Build with Conan <#building-with-conan>`_ to automate the process
   of getting dependencies or you can download and install the required
   dependencies manually and `Build with CMake <#building-with-cmake>`_
   directly.

.. note::
  Conan package manager automates the process of getting required packages
  so that you do not have to go to different web location and follow different
  instructions to install them.
