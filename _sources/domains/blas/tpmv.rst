.. _onemkl_blas_tpmv:

tpmv
====

Computes a matrix-vector product using a triangular packed matrix.

.. _onemkl_blas_tpmv_description:

.. rubric:: Description

The ``tpmv`` routines compute a matrix-vector product with a triangular
packed matrix. The operation is defined as:

.. math::

      x \leftarrow op(A)*x

where:

op(``A``) is one of op(``A``) = ``A``, or op(``A``) =
``A``\ :sup:`T`, or op(``A``) = ``A``\ :sup:`H`,

``A`` is an ``n``-by-``n`` unit or non-unit, upper or lower
triangular band matrix, supplied in packed form,

``x`` is a vector of length ``n``.

``tpmv`` supports the following precisions.

   .. list-table:: 
      :header-rows: 1

      * -  T 
      * -  ``float`` 
      * -  ``double`` 
      * -  ``std::complex<float>`` 
      * -  ``std::complex<double>`` 

.. _onemkl_blas_tpmv_buffer:

tpmv (Buffer Version)
---------------------

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::mkl::blas::column_major {
       void tpmv(sycl::queue &queue,
                 onemkl::uplo upper_lower,
                 onemkl::transpose trans,
                 onemkl::diag unit_nonunit,
                 std::int64_t n,
                 sycl::buffer<T,1> &a,
                 sycl::buffer<T,1> &x,
                 std::int64_t incx)
   }
.. code-block:: cpp

   namespace oneapi::mkl::blas::row_major {
       void tpmv(sycl::queue &queue,
                 onemkl::uplo upper_lower,
                 onemkl::transpose trans,
                 onemkl::diag unit_nonunit,
                 std::int64_t n,
                 sycl::buffer<T,1> &a,
                 sycl::buffer<T,1> &x,
                 std::int64_t incx)
   }

.. container:: section

   .. rubric:: Input Parameters

   queue
      The queue where the routine should be executed.

   upper_lower
      Specifies whether ``A`` is upper or lower triangular. See :ref:`onemkl_datatypes` for more details.

   trans
      Specifies op(``A``), the transposition operation applied to ``A``. See :ref:`onemkl_datatypes` for more details.

   unit_nonunit
      Specifies whether the matrix ``A`` is unit triangular or not. See :ref:`onemkl_datatypes` for more details.

   n
      Numbers of rows and columns of ``A``. Must be at least zero.

   a
      Buffer holding input matrix ``A``. Must have size at least
      (``n``\ \*(``n``\ +1))/2. See :ref:`matrix-storage` for
      more details.

   x
      Buffer holding input vector ``x``. The buffer must be of size at
      least (1 + (``n`` - 1)*abs(``incx``)). See :ref:`matrix-storage` for
      more details.

   incx
      Stride of vector ``x``.

.. container:: section

   .. rubric:: Output Parameters

   x
      Buffer holding the updated vector ``x``.


.. _onemkl_blas_tpmv_usm:

tpmv (USM Version)
------------------
      
.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::mkl::blas::column_major {
       sycl::event tpmv(sycl::queue &queue,
                        onemkl::uplo upper_lower,
                        onemkl::transpose trans,
                        onemkl::diag unit_nonunit,
                        std::int64_t n,
                        const T *a,
                        T *x,
                        std::int64_t incx,
                        const std::vector<sycl::event> &dependencies = {})
   }
.. code-block:: cpp

   namespace oneapi::mkl::blas::row_major {
       sycl::event tpmv(sycl::queue &queue,
                        onemkl::uplo upper_lower,
                        onemkl::transpose trans,
                        onemkl::diag unit_nonunit,
                        std::int64_t n,
                        const T *a,
                        T *x,
                        std::int64_t incx,
                        const std::vector<sycl::event> &dependencies = {})
   }

.. container:: section

   .. rubric:: Input Parameters

   queue
      The queue where the routine should be executed.

   upper_lower
      Specifies whether ``A`` is upper or lower triangular. See :ref:`onemkl_datatypes` for more details.

   trans
      Specifies op(``A``), the transposition operation applied to
      ``A``. See :ref:`onemkl_datatypes` for more details.

   unit_nonunit
      Specifies whether the matrix ``A`` is unit triangular or not. See :ref:`onemkl_datatypes` for more details.

   n
      Numbers of rows and columns of ``A``. Must be at least zero.

   a
      Pointer to input matrix ``A``. The array holding input matrix
      ``A`` must have size at least (``n``\ \*(``n``\ +1))/2. See
      :ref:`matrix-storage` for
      more details.

   x
      Pointer to input vector ``x``. The array holding input vector
      ``x`` must be of size at least (1 + (``n`` - 1)*abs(``incx``)).
      See :ref:`matrix-storage` for
      more details.

   incx
      Stride of vector ``x``.

   dependencies
      List of events to wait for before starting computation, if any.
      If omitted, defaults to no dependencies.

.. container:: section

   .. rubric:: Output Parameters

   x
      Pointer to the updated vector ``x``.

.. container:: section

   .. rubric:: Return Values

   Output event to wait on to ensure computation is complete.


   **Parent topic:** :ref:`blas-level-2-routines`
