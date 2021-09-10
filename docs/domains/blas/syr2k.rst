.. SPDX-FileCopyrightText: 2019-2020 Intel Corporation
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _onemkl_blas_syr2k:

syr2k
=====

Performs a symmetric rank-2k update.

.. _onemkl_blas_syr2k_description:

.. rubric:: Description

The ``syr2k`` routines perform a rank-2k update of an ``n`` x ``n``
symmetric matrix ``C`` by general matrices ``A`` and ``B``. 

If ``trans`` = ``transpose::nontrans``, the operation is defined as:

.. math::

      C \leftarrow alpha*(A*B^T + B*A^T) + beta*C

where ``A`` is ``n`` x ``k`` and ``B`` is ``k`` x ``n``.

If ``trans`` = ``transpose::trans``, the operation is defined as:

.. math::

      C \leftarrow alpha*(A^T*B + B^T*A) + beta * C


where ``A`` is ``k`` x ``n`` and ``B`` is ``n`` x ``k``.


In both cases:

``alpha`` and ``beta`` are scalars,

``C`` is a symmetric matrix and ``A``,\ ``B`` are general matrices,

The inner dimension of both matrix multiplications is ``k``.

``syr2k`` supports the following precisions:

   .. list-table:: 
      :header-rows: 1

      * -  T 
      * -  ``float`` 
      * -  ``double`` 
      * -  ``std::complex<float>`` 
      * -  ``std::complex<double>`` 

.. _onemkl_blas_syr2k_buffer:

syr2k (Buffer Version)
----------------------

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::mkl::blas::column_major {
       void syr2k(sycl::queue &queue,
                  onemkl::uplo upper_lower,
                  onemkl::transpose trans,
                  std::int64_t n,
                  std::int64_t k,
                  T alpha,
                  sycl::buffer<T,1> &a,
                  std::int64_t lda,
                  sycl::buffer<T,1> &b,
                  std::int64_t ldb,
                  T beta,
                  sycl::buffer<T,1> &c,
                  std::int64_t ldc)
   }
.. code-block:: cpp

   namespace oneapi::mkl::blas::row_major {
       void syr2k(sycl::queue &queue,
                  onemkl::uplo upper_lower,
                  onemkl::transpose trans,
                  std::int64_t n,
                  std::int64_t k,
                  T alpha,
                  sycl::buffer<T,1> &a,
                  std::int64_t lda,
                  sycl::buffer<T,1> &b,
                  std::int64_t ldb,
                  T beta,
                  sycl::buffer<T,1> &c,
                  std::int64_t ldc)
   }

.. container:: section

   .. rubric:: Input Parameters

   queue
      The queue where the routine should be executed.

   upper_lower
      Specifies whether ``A``'s data is stored in its upper or lower
      triangle. See :ref:`onemkl_datatypes` for more details.

   trans
      Specifies the operation to apply, as described above. Conjugation
      is never performed, even if ``trans`` = ``transpose::conjtrans``.

   n
      Number of rows and columns in ``C``.The value of ``n`` must be at
      least zero.

   k
      Inner dimension of matrix multiplications.The value of ``k`` must
      be at least zero.

   alpha
      Scaling factor for the rank-2k update.

   a
      Buffer holding input matrix ``A``.

      .. list-table::
         :header-rows: 1

         * -
           - ``trans`` = ``transpose::nontrans``
           - ``trans`` = ``transpose::trans`` or ``transpose::conjtrans``
         * - Column major
           - ``A`` is an ``n``-by-``k`` matrix so the array ``a``
             must have size at least ``lda``\ \*\ ``k``.
           - ``A`` is an ``k``-by-``n`` matrix so the array ``a``
             must have size at least ``lda``\ \*\ ``n``
         * - Row major
           - ``A`` is an ``n``-by-``k`` matrix so the array ``a``
             must have size at least ``lda``\ \*\ ``n``.
           - ``A`` is an ``k``-by-``n`` matrix so the array ``a``
             must have size at least ``lda``\ \*\ ``k``.

      See :ref:`matrix-storage` for
      more details.

   lda
      The leading dimension of ``A``. It must be positive.

      .. list-table::
         :header-rows: 1

         * -
           - ``trans`` = ``transpose::nontrans``
           - ``trans`` = ``transpose::trans`` or ``transpose::conjtrans``
         * - Column major
           - ``lda`` must be at least ``n``.
           - ``lda`` must be at least ``k``.
         * - Row major
           - ``lda`` must be at least ``k``.
           - ``lda`` must be at least ``n``.

   b
      Buffer holding input matrix ``B``.

      .. list-table::
         :header-rows: 1

         * -
           - ``trans`` = ``transpose::nontrans``
           - ``trans`` = ``transpose::trans`` or ``transpose::conjtrans``
         * - Column major
           - ``B`` is an ``k``-by-``n`` matrix so the array ``b``
             must have size at least ``ldb``\ \*\ ``n``.
           - ``B`` is an ``n``-by-``k`` matrix so the array ``b``
             must have size at least ``ldb``\ \*\ ``k``
         * - Row major
           - ``B`` is an ``k``-by-``n`` matrix so the array ``b``
             must have size at least ``ldb``\ \*\ ``k``.
           - ``B`` is an ``n``-by-``k`` matrix so the array ``b``
             must have size at least ``ldb``\ \*\ ``n``.

      See :ref:`matrix-storage`
      for more details.

   ldb
      The leading dimension of ``B``. It must be positive.

      .. list-table::
         :header-rows: 1

         * -
           - ``trans`` = ``transpose::nontrans``
           - ``trans`` = ``transpose::trans`` or ``transpose::conjtrans``
         * - Column major
           - ``ldb`` must be at least ``k``.
           - ``ldb`` must be at least ``n``.
         * - Row major
           - ``ldb`` must be at least ``n``.
           - ``ldb`` must be at least ``k``.

   beta
      Scaling factor for matrix ``C``.

   c
      Buffer holding input/output matrix ``C``. Must have size at least
      ``ldc``\ \*\ ``n``. See :ref:`matrix-storage` for
      more details

   ldc
      Leading dimension of ``C``. Must be positive and at least ``n``.

.. container:: section

   .. rubric:: Output Parameters

   c
      Output buffer, overwritten by the updated ``C`` matrix.

.. container:: section

   .. rubric:: Throws

   This routine shall throw the following exceptions if the associated condition is detected. An implementation may throw additional implementation-specific exception(s) in case of error conditions not covered here.

   :ref:`oneapi::mkl::invalid_argument<onemkl_exception_invalid_argument>`
       
   
   :ref:`oneapi::mkl::unsupported_device<onemkl_exception_unsupported_device>`
       

   :ref:`oneapi::mkl::host_bad_alloc<onemkl_exception_host_bad_alloc>`
       

   :ref:`oneapi::mkl::device_bad_alloc<onemkl_exception_device_bad_alloc>`
       

   :ref:`oneapi::mkl::unimplemented<onemkl_exception_unimplemented>`
      

.. _onemkl_blas_syr2k_usm:

syr2k (USM Version)
-------------------

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::mkl::blas::column_major {
       sycl::event syr2k(sycl::queue &queue,
                         onemkl::uplo upper_lower,
                         onemkl::transpose trans,
                         std::int64_t n,
                         std::int64_t k,
                         T alpha,
                         const T* a,
                         std::int64_t lda,
                         const T* b,
                         std::int64_t ldb,
                         T beta,
                         T* c,
                         std::int64_t ldc,
                         const sycl::vector_class<sycl::event> &dependencies = {})
   }
.. code-block:: cpp

   namespace oneapi::mkl::blas::row_major {
       sycl::event syr2k(sycl::queue &queue,
                         onemkl::uplo upper_lower,
                         onemkl::transpose trans,
                         std::int64_t n,
                         std::int64_t k,
                         T alpha,
                         const T* a,
                         std::int64_t lda,
                         const T* b,
                         std::int64_t ldb,
                         T beta,
                         T* c,
                         std::int64_t ldc,
                         const sycl::vector_class<sycl::event> &dependencies = {})
   }

.. container:: section

   .. rubric:: Input Parameters

   queue
      The queue where the routine should be executed.

   upper_lower
      Specifies whether ``A``'s data is stored in its upper or lower
      triangle. See :ref:`onemkl_datatypes` for more details.

   trans
      Specifies the operation to apply, as described above.
      Conjugation is never performed, even if ``trans`` =
      ``transpose::conjtrans``.

   n
      Number of rows and columns in ``C``. The value of ``n`` must be
      at least zero.

   k
      Inner dimension of matrix multiplications.The value of ``k``
      must be at least zero.

   alpha
      Scaling factor for the rank-2k update.

   a
      Pointer to input matrix ``A``.

      .. list-table::
         :header-rows: 1

         * -
           - ``trans`` = ``transpose::nontrans``
           - ``trans`` = ``transpose::trans`` or ``transpose::conjtrans``
         * - Column major
           - ``A`` is an ``n``-by-``k`` matrix so the array ``a``
             must have size at least ``lda``\ \*\ ``k``.
           - ``A`` is an ``k``-by-``n`` matrix so the array ``a``
             must have size at least ``lda``\ \*\ ``n``
         * - Row major
           - ``A`` is an ``n``-by-``k`` matrix so the array ``a``
             must have size at least ``lda``\ \*\ ``n``.
           - ``A`` is an ``k``-by-``n`` matrix so the array ``a``
             must have size at least ``lda``\ \*\ ``k``.
      
      See :ref:`matrix-storage` for more details.

   lda
      The leading dimension of ``A``. It must be positive.

      .. list-table::
         :header-rows: 1

         * -
           - ``trans`` = ``transpose::nontrans``
           - ``trans`` = ``transpose::trans`` or ``transpose::conjtrans``
         * - Column major
           - ``lda`` must be at least ``n``.
           - ``lda`` must be at least ``k``.
         * - Row major
           - ``lda`` must be at least ``k``.
           - ``lda`` must be at least ``n``.

   b
      Pointer to input matrix ``B``.

      .. list-table::
         :header-rows: 1

         * -
           - ``trans`` = ``transpose::nontrans``
           - ``trans`` = ``transpose::trans`` or ``transpose::conjtrans``
         * - Column major
           - ``B`` is an ``k``-by-``n`` matrix so the array ``b``
             must have size at least ``ldb``\ \*\ ``n``.
           - ``B`` is an ``n``-by-``k`` matrix so the array ``b``
             must have size at least ``ldb``\ \*\ ``k``
         * - Row major
           - ``B`` is an ``k``-by-``n`` matrix so the array ``b``
             must have size at least ``ldb``\ \*\ ``k``.
           - ``B`` is an ``n``-by-``k`` matrix so the array ``b``
             must have size at least ``ldb``\ \*\ ``n``.
   
      See :ref:`matrix-storage` for
      more details.

   ldb
      The leading dimension of ``B``. It must be positive.

      .. list-table::
         :header-rows: 1

         * -
           - ``trans`` = ``transpose::nontrans``
           - ``trans`` = ``transpose::trans`` or ``transpose::conjtrans``
         * - Column major
           - ``ldb`` must be at least ``k``.
           - ``ldb`` must be at least ``n``.
         * - Row major
           - ``ldb`` must be at least ``n``.
           - ``ldb`` must be at least ``k``.

   beta
      Scaling factor for matrix ``C``.

   c
      Pointer to input/output matrix ``C``. Must have size at least
      ``ldc``\ \*\ ``n``. See :ref:`matrix-storage` for
      more details

   ldc
      Leading dimension of ``C``. Must be positive and at least
      ``n``.

   dependencies
      List of events to wait for before starting computation, if any.
      If omitted, defaults to no dependencies.

.. container:: section

   .. rubric:: Output Parameters

   c
      Pointer to the output matrix, overwritten by the updated ``C``
      matrix.

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
      

   **Parent topic:** :ref:`blas-level-3-routines`
