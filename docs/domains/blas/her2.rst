.. _onemkl_blas_her2:

her2
====

Computes a rank-2 update of a Hermitian matrix.

.. _onemkl_blas_her2_description:

.. rubric:: Description

The ``her2`` routines compute two scalar-vector-vector products and add
them to a Hermitian matrix. The operation is defined as:

.. math::

      A \leftarrow alpha*x*y^H + conjg(alpha)*y*x^H + A

where:

``alpha`` is a scalar,

``A`` is an ``n``-by-``n`` Hermitian matrix,

``x`` and ``y`` are vectors or length ``n``.

``her2`` supports the following precisions.

   .. list-table:: 
      :header-rows: 1

      * -  T 
      * -  ``std::complex<float>`` 
      * -  ``std::complex<double>`` 

.. _onemkl_blas_her2_buffer:

her2 (Buffer Version)
---------------------

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::mkl::blas::column_major {
       void her2(sycl::queue &queue,
                 onemkl::uplo upper_lower,
                 std::int64_t n,
                 T alpha,
                 sycl::buffer<T,1> &x,
                 std::int64_t incx,
                 sycl::buffer<T,1> &y,
                 std::int64_t incy,
                 sycl::buffer<T,1> &a,
                 std::int64_t lda)
   }
.. code-block:: cpp

   namespace oneapi::mkl::blas::row_major {
       void her2(sycl::queue &queue,
                 onemkl::uplo upper_lower,
                 std::int64_t n,
                 T alpha,
                 sycl::buffer<T,1> &x,
                 std::int64_t incx,
                 sycl::buffer<T,1> &y,
                 std::int64_t incy,
                 sycl::buffer<T,1> &a,
                 std::int64_t lda)
   }

.. container:: section

   .. rubric:: Input Parameters

   queue
      The queue where the routine should be executed.

   upper_lower
      Specifies whether ``A`` is upper or lower triangular. See :ref:`onemkl_datatypes` for more details.

   n
      Number of columns of ``A``. Must be at least zero.

   alpha
      Scaling factor for the matrix-vector product.

   x
      Buffer holding input vector ``x``. The buffer must be of size at
      least (1 + (``n`` - 1)*abs(``incx``)). See :ref:`matrix-storage` for
      more details.

   incx
      Stride of vector ``x``.

   y
      Buffer holding input/output vector ``y``. The buffer must be of
      size at least (1 + (``n`` - 1)*abs(``incy``)). See :ref:`matrix-storage`
      for more details.

   incy
      Stride of vector ``y``.

   a
      Buffer holding input matrix ``A``. Must have size at least
      ``lda``\ \*\ ``n``. See :ref:`matrix-storage` for
      more details.

   lda
      Leading dimension of matrix ``A``. Must be at least ``n``, and
      positive.

.. container:: section

   .. rubric:: Output Parameters

   a
      Buffer holding the updated upper triangular part of the Hermitian
      matrix ``A`` if ``upper_lower``\ \=\ ``upper``, or the updated
      lower triangular part of the Hermitian matrix ``A`` if
      ``upper_lower``\ \=\ ``lower``.

      The imaginary parts of the diagonal elements are set to zero.

      

.. _onemkl_blas_her2_usm:

her2 (USM Version)
------------------

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::mkl::blas::column_major {
       sycl::event her2(sycl::queue &queue,
                        onemkl::uplo upper_lower,
                        std::int64_t n,
                        T alpha,
                        const T *x,
                        std::int64_t incx,
                        const T *y,
                        std::int64_t incy,
                        T *a,
                        std::int64_t lda,
                        const std::vector<sycl::event> &dependencies = {})
   }
.. code-block:: cpp

   namespace oneapi::mkl::blas::row_major {
       sycl::event her2(sycl::queue &queue,
                        onemkl::uplo upper_lower,
                        std::int64_t n,
                        T alpha,
                        const T *x,
                        std::int64_t incx,
                        const T *y,
                        std::int64_t incy,
                        T *a,
                        std::int64_t lda,
                        const std::vector<sycl::event> &dependencies = {})
   }

.. container:: section

   .. rubric:: Input Parameters

   queue
      The queue where the routine should be executed.

   upper_lower
      Specifies whether ``A`` is upper or lower triangular. See :ref:`onemkl_datatypes` for more details.

   n
      Number of columns of ``A``. Must be at least zero.

   alpha
      Scaling factor for the matrix-vector product.

   x
      Pointer to input vector ``x``. The array holding input vector
      ``x`` must be of size at least (1 + (``n`` - 1)*abs(``incx``)).
      See :ref:`matrix-storage` for
      more details.

   incx
      Stride of vector ``x``.

   y
      Pointer to input/output vector ``y``. The array holding
      input/output vector ``y`` must be of size at least (1 + (``n``
      - 1)*abs(``incy``)). See :ref:`matrix-storage` for
      more details.

   incy
      Stride of vector ``y``.

   a
      Pointer to input matrix ``A``. The array holding input matrix
      ``A`` must have size at least ``lda``\ \*\ ``n``. See :ref:`matrix-storage` for
      more details.

   lda
      Leading dimension of matrix ``A``. Must be at least ``n``, and
      positive.

   dependencies
      List of events to wait for before starting computation, if any.
      If omitted, defaults to no dependencies.

.. container:: section

   .. rubric:: Output Parameters

   a
      Pointer to the updated upper triangular part of the Hermitian
      matrix ``A`` if ``upper_lower``\ \=\ ``upper``, or the updated
      lower triangular part of the Hermitian matrix ``A`` if
      ``upper_lower``\ \=\ ``lower``.

      The imaginary parts of the diagonal elements are set to zero.

.. container:: section

   .. rubric:: Return Values

   Output event to wait on to ensure computation is complete.


   **Parent topic:** :ref:`blas-level-2-routines`
