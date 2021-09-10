.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemkl_blas_spr:

spr
===

Performs a rank-1 update of a symmetric packed matrix.

.. _onemkl_blas_spr_description:

.. rubric:: Description

The ``spr`` routines compute a scalar-vector-vector product and add the
result to a symmetric packed matrix. The operation is defined as:

.. math::

      A \leftarrow alpha*x*x^T + A

where:

``alpha`` is scalar,

``A`` is an ``n``-by-``n`` symmetric matrix, supplied in packed form,

``x`` is a vector of length ``n``.

``spr`` supports the following precisions.

   .. list-table:: 
      :header-rows: 1

      * -  T 
      * -  ``float`` 
      * -  ``double`` 

.. _onemkl_blas_spr_buffer:

spr (Buffer Version)
--------------------

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::mkl::blas::column_major {
       void spr(sycl::queue &queue,
                onemkl::uplo upper_lower,
                std::std::int64_t n,
                T alpha,
                sycl::buffer<T,1> &x,
                std::int64_t incx,
                sycl::buffer<T,1> &a)
   }
.. code-block:: cpp

   namespace oneapi::mkl::blas::row_major {
       void spr(sycl::queue &queue,
                onemkl::uplo upper_lower,
                std::std::int64_t n,
                T alpha,
                sycl::buffer<T,1> &x,
                std::int64_t incx,
                sycl::buffer<T,1> &a)
   }

.. container:: section

   .. rubric:: Input Parameters

   queue
      The queue where the routine should be executed.

   upper_lower
      Specifies whether ``A`` is upper or lower triangular. See :ref:`onemkl_datatypes` for more details.

   n
      Number of rows and columns of ``A``. Must be at least zero.

   alpha
      Scaling factor for the matrix-vector product.

   x
      Buffer holding input vector ``x``. The buffer must be of size at
      least (1 + (``n`` - 1)*abs(``incx``)). See :ref:`matrix-storage` for
      more details.

   incx
      Stride of vector ``x``.

   a
      Buffer holding input matrix ``A``. Must have size at least
      (``n``\ \*(``n`` + 1))/2. See :ref:`matrix-storage` for
      more details.

.. container:: section

   .. rubric:: Output Parameters
      :class: sectiontitle

   a
      Buffer holding the updated upper triangular part of the symmetric
      matrix ``A`` if ``upper_lower``\ \=\ ``upper``, or the updated lower
      triangular part of the symmetric matrix ``A`` if
      ``upper_lower``\ \=\ ``lower``.

.. container:: section

   .. rubric:: Throws

   This routine shall throw the following exceptions if the associated condition is detected. An implementation may throw additional implementation-specific exception(s) in case of error conditions not covered here.

   :ref:`oneapi::mkl::invalid_argument<onemkl_exception_invalid_argument>`
       
   
   :ref:`oneapi::mkl::unsupported_device<onemkl_exception_unsupported_device>`
       

   :ref:`oneapi::mkl::host_bad_alloc<onemkl_exception_host_bad_alloc>`
       

   :ref:`oneapi::mkl::device_bad_alloc<onemkl_exception_device_bad_alloc>`
       

   :ref:`oneapi::mkl::unimplemented<onemkl_exception_unimplemented>`
      

.. _onemkl_blas_spr_usm:

spr (USM Version)
-----------------

.. rubric:: Syntax
         
.. code-block:: cpp

   namespace oneapi::mkl::blas::column_major {
       sycl::event spr(sycl::queue &queue,
                       onemkl::uplo upper_lower,
                       std::int64_t n,
                       T alpha,
                       const T *x,
                       std::int64_t incx,
                       T *a,
                       const sycl::vector_class<sycl::event> &dependencies = {})
   }
.. code-block:: cpp

   namespace oneapi::mkl::blas::row_major {
       sycl::event spr(sycl::queue &queue,
                       onemkl::uplo upper_lower,
                       std::int64_t n,
                       T alpha,
                       const T *x,
                       std::int64_t incx,
                       T *a,
                       const sycl::vector_class<sycl::event> &dependencies = {})
   }
   
.. container:: section

   .. rubric:: Input Parameters

   queue
      The queue where the routine should be executed.

   upper_lower
      Specifies whether ``A`` is upper or lower triangular. See :ref:`onemkl_datatypes` for more details.

   n
      Number of rows and columns of ``A``. Must be at least zero.

   alpha
      Scaling factor for the matrix-vector product.

   x
      Pointer to input vector ``x``. The array holding input vector
      ``x`` must be of size at least (1 + (``n`` - 1)*abs(``incx``)).
      See :ref:`matrix-storage` for
      more details.

   incx
      Stride of vector ``x``.

   a
      Pointer to input matrix ``A``. The array holding input matrix
      ``A`` must have size at least (``n``\ \*(``n`` + 1))/2. See
      :ref:`matrix-storage` for
      more details.

   dependencies
      List of events to wait for before starting computation, if any.
      If omitted, defaults to no dependencies.

.. container:: section

   .. rubric:: Output Parameters

   a
      Pointer to the updated upper triangular part of the symmetric
      matrix ``A`` if ``upper_lower``\ \=\ ``upper``, or the updated lower
      triangular part of the symmetric matrix ``A`` if
      ``upper_lower``\ \=\ ``lower``.

.. container:: section

   .. rubric:: Return Values

   Output event to wait on to ensure computation is complete.

.. container:: section

   .. rubric:: Throws

   This routine shall throw the following exceptions if the associated condition is detected. An implementation may throw additional implementation-specific exception(s) in case of error conditions not covered here.

   :ref:`oneapi::mkl::invalid_argument<onemkl_exception_invalid_argument>`
       
       
   
   :ref:`oneapi::mkl::unsupported_device<onemkl_exception_unsupported_device>`
       

   :ref:`oneapi::mkl::host_bad_alloc<onemkl_exception_host_bad_alloc>`
       

   :ref:`oneapi::mkl::device_bad_alloc<onemkl_exception_device_bad_alloc>`
       

   :ref:`oneapi::mkl::unimplemented<onemkl_exception_unimplemented>`
      

   **Parent topic:** :ref:`blas-level-2-routines`
