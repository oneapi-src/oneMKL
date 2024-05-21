.. _onemkl_blas_syr:

syr
===

Computes a rank-1 update of a symmetric matrix.

.. _onemkl_blas_syr_description:

.. rubric:: Description

The ``syr`` routines compute a scalar-vector-vector product add them and
add the result to a matrix, with a symmetric matrix. The operation is
defined as:

.. math::

      A \leftarrow alpha*x*x^T + A

where:

``alpha`` is scalar,

``A`` is an ``n``-by-``n`` symmetric matrix,

``x`` is a vector of length ``n``.

``syr`` supports the following precisions.

   .. list-table:: 
      :header-rows: 1

      * -  T 
      * -  ``float`` 
      * -  ``double`` 

.. _onemkl_blas_syr_buffer:

syr (Buffer Version)
--------------------

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::mkl::blas::column_major {
       void syr(sycl::queue &queue,
                onemkl::uplo upper_lower,
                std::int64_t n,
                T alpha,
                sycl::buffer<T,1> &x,
                std::int64_t incx,
                sycl::buffer<T,1> &a,
                std::int64_t lda)
   }
.. code-block:: cpp

   namespace oneapi::mkl::blas::row_major {
       void syr(sycl::queue &queue,
                onemkl::uplo upper_lower,
                std::int64_t n,
                T alpha,
                sycl::buffer<T,1> &x,
                std::int64_t incx,
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
      Buffer holding the updated upper triangular part of the symmetric
      matrix ``A`` if ``upper_lower``\ \=\ ``upper`` or the updated lower
      triangular part of the symmetric matrix ``A`` if
      ``upper_lower``\ \=\ ``lower``.


.. _onemkl_blas_syr_usm:

syr (USM Version)
-----------------

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::mkl::blas::column_major {
       sycl::event syr(sycl::queue &queue,
                       onemkl::uplo upper_lower,
                       std::int64_t n,
                       T alpha,
                       const T *x,
                       std::int64_t incx,
                       T *a,
                       std::int64_t lda,
                       const std::vector<sycl::event> &dependencies = {})
   }
.. code-block:: cpp

   namespace oneapi::mkl::blas::row_major {
       sycl::event syr(sycl::queue &queue,
                       onemkl::uplo upper_lower,
                       std::int64_t n,
                       T alpha,
                       const T *x,
                       std::int64_t incx,
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
      Pointer to the updated upper triangular part of the symmetric
      matrix ``A`` if ``upper_lower``\ \=\ ``upper`` or the updated lower
      triangular part of the symmetric matrix ``A`` if
      ``upper_lower``\ \=\ ``lower``.

.. container:: section

   .. rubric:: Return Values

   Output event to wait on to ensure computation is complete.


   **Parent topic:** :ref:`blas-level-2-routines`
