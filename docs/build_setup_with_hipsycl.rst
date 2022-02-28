.. _build_setup_with_hipsycl:

Build Setup with hipSYCL
========================

#. 
   Make sure that the dependencies of hipSYCL are fulfilled. For a detailed
   description, see the
   `hipSYCL installation readme <https://github.com/illuhad/hipSYCL/blob/develop/doc/installing.md#software-dependencies>`_.

#. 
   Install hipSYCL with the prefered backends enabled. hipSYCL supports
   various backends. You can customize support for the target system at
   compile time by setting the appropriate configuration flags; see the
   `hipSYCL documentation <https://github.com/illuhad/hipSYCL/blob/develop/doc/installing.md>`_
   for instructions.

#. 
   Install `AMD rocBLAS <https://rocblas.readthedocs.io/en/master/install.html>`_.

#. 
   Clone this project to ``<path to onemkl>``, where ``<path to onemkl>`` is
   the root directory of this repository.

#. 
   Download and install the required dependencies manually and
   :ref:`Build with CMake <building_with_cmake>`.

