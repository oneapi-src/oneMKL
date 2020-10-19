.. _onemkl_blas_trsm_batch:

trsm_batch
==========

Computes a group of ``trsm`` operations.

.. _onemkl_blas_trsm_batch_description:

.. rubric:: Description

The ``trsm_batch`` routines are batched versions of :ref:`onemkl_blas_trsm`, performing
multiple ``trsm`` operations in a single call. Each ``trsm`` 
solves an equation of the form op(A) \* X = alpha \* B or X \* op(A) = alpha \* B. 
   
``trsm_batch`` supports the following precisions.

   .. list-table:: 
      :header-rows: 1

      * -  T 
      * -  ``float`` 
      * -  ``double`` 
      * -  ``std::complex<float>`` 
      * -  ``std::complex<double>`` 

.. _onemkl_blas_trsm_batch_buffer:

trsm_batch (Buffer Version)
---------------------------

.. rubric:: Description

The buffer version of ``trsm_batch`` supports only the strided API. 
   
The strided API operation is defined as:
::

   for i = 0 … batch_size – 1
       A and B are matrices at offset i * stridea and i * strideb in a and b.
       if (left_right == onemkl::side::left) then
           compute X such that op(A) * X = alpha * B
       else
           compute X such that X * op(A) = alpha * B
       end if
       B := X
   end for

where:

op(``A``) is one of op(``A``) = ``A``, or op(A) = ``A``\ :sup:`T`,
or op(``A``) = ``A``\ :sup:`H`,

``alpha`` is a scalar,

``A`` is a triangular matrix,

``B`` and ``X`` are ``m`` x ``n`` general matrices,

``A`` is either ``m`` x ``m`` or ``n`` x ``n``,depending on whether
it multiplies ``X`` on the left or right. On return, the matrix ``B``
is overwritten by the solution matrix ``X``.

The ``a`` and ``b`` buffers contain all the input matrices. The stride 
between matrices is given by the stride parameter. The total number
of matrices in ``a`` and ``b`` buffers are given by the ``batch_size`` parameter.

**Strided API**

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::mkl::blas::column_major {
       void trsm_batch(sycl::queue &queue,
                       onemkl::side left_right,
                       onemkl::uplo upper_lower,
                       onemkl::transpose trans,
                       onemkl::diag unit_diag,
                       std::int64_t m,
                       std::int64_t n,
                       T alpha,
                       sycl::buffer<T,1> &a,
                       std::int64_t lda,
                       std::int64_t stridea,
                       sycl::buffer<T,1> &b,
                       std::int64_t ldb,
                       std::int64_t strideb,
                       std::int64_t batch_size)
   }
.. code-block:: cpp

   namespace oneapi::mkl::blas::row_major {
       void trsm_batch(sycl::queue &queue,
                       onemkl::side left_right,
                       onemkl::uplo upper_lower,
                       onemkl::transpose trans,
                       onemkl::diag unit_diag,
                       std::int64_t m,
                       std::int64_t n,
                       T alpha,
                       sycl::buffer<T,1> &a,
                       std::int64_t lda,
                       std::int64_t stridea,
                       sycl::buffer<T,1> &b,
                       std::int64_t ldb,
                       std::int64_t strideb,
                       std::int64_t batch_size)
   }

.. container:: section

   .. rubric:: Input Parameters

   queue
      The queue where the routine should be executed.

   left_right
      Specifies whether the matrices ``A`` multiply ``X`` on the left
      (``side::left``) or on the right (``side::right``). See :ref:`onemkl_datatypes` for more details.

   upper_lower
      Specifies whether the matrices ``A`` are upper or lower
      triangular. See :ref:`onemkl_datatypes` for more details.

   trans
      Specifies op(``A``), the transposition operation applied to the
      matrices ``A``. See :ref:`onemkl_datatypes` for more details.

   unit_diag
      Specifies whether the matrices ``A`` are assumed to be unit
      triangular (all diagonal elements are 1). See :ref:`onemkl_datatypes` for more details.

   m
      Number of rows of the ``B`` matrices. Must be at least zero.

   n
      Number of columns of the ``B`` matrices. Must be at least zero.

   alpha
      Scaling factor for the solutions.

   a
      Buffer holding the input matrices ``A`` with size ``stridea`` * ``batch_size``.

   lda
      Leading dimension of the matrices ``A``. Must be at least ``m`` if
      ``left_right`` = ``side::left``, and at least ``n`` if ``left_right`` =
      ``side::right``. Must be positive.

   stridea
      Stride between different ``A`` matrices.

   b
      Buffer holding the input matrices ``B`` with size ``strideb`` * ``batch_size``.

   ldb
      Leading dimension of the matrices ``B``. It must be positive and at least
      ``m`` if column major layout is used to store matrices or at
      least ``n`` if row major layout is used to store matrices.

   strideb
      Stride between different ``B`` matrices.

   batch_size
      Specifies the number of triangular linear systems to solve.

.. container:: section

   .. rubric:: Output Parameters

   b
      Output buffer, overwritten by ``batch_size`` solution matrices
      ``X``.

.. container:: section

   .. rubric:: Notes

   If ``alpha`` = 0, matrix ``B`` is set to zero and the matrices ``A``
   and ``B`` do not need to be initialized before calling ``trsm_batch``.

   **Parent topic:** :ref:`blas-like-extensions`
