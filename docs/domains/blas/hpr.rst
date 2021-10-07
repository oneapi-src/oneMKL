.. _onemkl_blas_hpr:

hpr
===

Computes a rank-1 update of a Hermitian packed matrix.

.. _onemkl_blas_hpr_description:

.. rubric:: Description

The ``hpr`` routines compute a scalar-vector-vector product and add the
result to a Hermitian packed matrix. The operation is defined as

.. math::

      A \leftarrow alpha*x*x^H + A

where:

``alpha`` is scalar,

``A`` is an ``n``-by-``n`` Hermitian matrix, supplied in packed form,

``x`` is a vector of length ``n``.

``hpr`` supports the following precisions.

   .. list-table:: 
      :header-rows: 1

      * -  T 
      * -  ``std::complex<float>`` 
      * -  ``std::complex<double>`` 

.. _onemkl_blas_hpr_buffer:

hpr (Buffer Version)
--------------------

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::mkl::blas::column_major {
       void hpr(sycl::queue &queue,
                onemkl::uplo upper_lower,
                std::int64_t n,
                T alpha,
                sycl::buffer<T,1> &x,
                std::int64_t incx,
                sycl::buffer<T,1> &a)
   }
.. code-block:: cpp

   namespace oneapi::mkl::blas::row_major {
       void hpr(sycl::queue &queue,
                onemkl::uplo upper_lower,
                std::int64_t n,
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
      (``n``\ \*(``n``-1))/2. See :ref:`matrix-storage` for
      more details.

      The imaginary part of the diagonal elements need not be set and
      are assumed to be zero.

.. container:: section

   .. rubric:: Output Parameters

   a
      Buffer holding the updated upper triangular part of the Hermitian
      matrix ``A`` if ``upper_lower``\ \=\ ``upper``, or the updated lower
      triangular part of the Hermitian matrix ``A`` if
      ``upper_lower``\ \=\ ``lower``.

      The imaginary parts of the diagonal elements are set to zero.


.. _onemkl_blas_hpr_usm:

hpr (USM Version)
-----------------

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::mkl::blas::column_major {
       sycl::event hpr(sycl::queue &queue,
                       onemkl::uplo upper_lower,
                       std::int64_t n,
                       T alpha,
                       const T *x,
                       std::int64_t incx,
                       T *a,
                       const std::vector<sycl::event> &dependencies = {})
   }
.. code-block:: cpp

   namespace oneapi::mkl::blas::row_major {
       sycl::event hpr(sycl::queue &queue,
                       onemkl::uplo upper_lower,
                       std::int64_t n,
                       T alpha,
                       const T *x,
                       std::int64_t incx,
                       T *a,
                       const std::vector<sycl::event> &dependencies = {})
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
      ``A`` must have size at least (``n``\ \*(``n``-1))/2. See
      :ref:`matrix-storage` for
      more details.

      The imaginary part of the diagonal elements need not be set and
      are assumed to be zero.

   dependencies
      List of events to wait for before starting computation, if any.
      If omitted, defaults to no dependencies.

.. container:: section

   .. rubric:: Output Parameters

   a
      Pointer to the updated upper triangular part of the Hermitian
      matrix ``A`` if ``upper_lower``\ \=\ ``upper``, or the updated lower
      triangular part of the Hermitian matrix ``A`` if
      ``upper_lower``\ \=\ ``lower``.

      The imaginary parts of the diagonal elements are set to zero.

.. container:: section

   .. rubric:: Return Values

   Output event to wait on to ensure computation is complete.

      

   **Parent topic:** :ref:`blas-level-2-routines`
