.. _onemkl_blas_trsm:

trsm
====

Solves a triangular matrix equation (forward or backward solve).

.. _onemkl_blas_trsm_description:

.. rubric:: Description

The ``trsm`` routines solve one of the following matrix equations:

.. math::

      op(A)*X = alpha*B

or

.. math::

      X*op(A) = alpha*B

where:

op(``A``) is one of op(``A``) = ``A``, or op(``A``) =
``A``\ :sup:`T`, or op(``A``) = ``A``\ :sup:`H`,

``alpha`` is a scalar,

``A`` is a triangular matrix, and

``B`` and ``X`` are ``m`` x ``n`` general matrices.

``A`` is either ``m`` x ``m`` or ``n`` x ``n``, depending on whether
it multiplies ``X`` on the left or right. On return, the matrix ``B``
is overwritten by the solution matrix ``X``.

``trsm`` supports the following precisions.

   .. list-table:: 
      :header-rows: 1

      * -  T 
      * -  ``float`` 
      * -  ``double`` 
      * -  ``std::complex<float>`` 
      * -  ``std::complex<double>`` 

.. _onemkl_blas_trsm_buffer:

trsm (Buffer Version)
---------------------

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::mkl::blas::column_major {
       void trsm(sycl::queue &queue,
                 onemkl::side left_right,
                 onemkl::uplo upper_lower,
                 onemkl::transpose transa,
                 onemkl::diag unit_diag,
                 std::int64_t m,
                 std::int64_t n,
                 T alpha,
                 sycl::buffer<T,1> &a,
                 std::int64_t lda,
                 sycl::buffer<T,1> &b,
                 std::int64_t ldb)
   }
.. code-block:: cpp

   namespace oneapi::mkl::blas::row_major {
       void trsm(sycl::queue &queue,
                 onemkl::side left_right,
                 onemkl::uplo upper_lower,
                 onemkl::transpose transa,
                 onemkl::diag unit_diag,
                 std::int64_t m,
                 std::int64_t n,
                 T alpha,
                 sycl::buffer<T,1> &a,
                 std::int64_t lda,
                 sycl::buffer<T,1> &b,
                 std::int64_t ldb)
   }

.. container:: section

   .. rubric:: Input Parameters

   queue
      The queue where the routine should be executed.

   left_right
      Specifies whether ``A`` multiplies ``X`` on the left
      (``side::left``) or on the right (``side::right``). See :ref:`onemkl_datatypes` for more details.

   uplo
      Specifies whether the matrix ``A`` is upper or lower triangular. See :ref:`onemkl_datatypes` for more details.

   trans
      Specifies op(``A``), the transposition operation applied to ``A``. See :ref:`onemkl_datatypes` for more details.

   unit_diag
      Specifies whether ``A`` is assumed to be unit triangular (all
      diagonal elements are 1). See :ref:`onemkl_datatypes` for more details.

   m
      Specifies the number of rows of ``B``. The value of ``m`` must be
      at least zero.

   n
      Specifies the number of columns of ``B``. The value of ``n`` must
      be at least zero.

   alpha
      Scaling factor for the solution.

   a
      Buffer holding input matrix ``A``. Must have size at least
      ``lda``\ \*\ ``m`` if ``left_right`` = ``side::left``, or
      ``lda``\ \*\ ``n`` if ``left_right`` = ``side::right``. See
      :ref:`matrix-storage` for
      more details.

   lda
      Leading dimension of ``A``. Must be at least ``m`` if
      ``left_right`` = ``side::left``, and at least ``n`` if
      ``left_right`` = ``side::right``. Must be positive.

   b
      Buffer holding input/output matrix ``B``. Must have size at
      least ``ldb``\ \*\ ``n`` if column major layout is used to store
      matrices or at least ``ldb``\ \*\ ``m`` if row major layout is
      used to store matrices. See :ref:`matrix-storage` for more details.

   ldb
      Leading dimension of ``B``. It must be positive and at least
      ``m`` if column major layout is used to store matrices or at
      least ``n`` if row major layout is used to store matrices.

.. container:: section

   .. rubric:: Output Parameters

   b
      Output buffer. Overwritten by the solution matrix ``X``.

.. container:: section

   .. rubric:: Notes

   If ``alpha`` = 0, matrix ``B`` is set to zero, and ``A`` and ``B`` do
   not need to be initialized at entry.

      

.. _onemkl_blas_trsm_usm:

trsm (USM Version)
------------------

.. rubric:: Syntax

.. code-block:: cpp

   namespace oneapi::mkl::blas::column_major {
       sycl::event trsm(sycl::queue &queue,
                        onemkl::side left_right,
                        onemkl::uplo upper_lower,
                        onemkl::transpose transa,
                        onemkl::diag unit_diag,
                        std::int64_t m,
                        std::int64_t n,
                        T alpha,
                        const T* a,
                        std::int64_t lda,
                        T* b,
                        std::int64_t ldb,
                        const std::vector<sycl::event> &dependencies = {})
   }
.. code-block:: cpp

   namespace oneapi::mkl::blas::row_major {
       sycl::event trsm(sycl::queue &queue,
                        onemkl::side left_right,
                        onemkl::uplo upper_lower,
                        onemkl::transpose transa,
                        onemkl::diag unit_diag,
                        std::int64_t m,
                        std::int64_t n,
                        T alpha,
                        const T* a,
                        std::int64_t lda,
                        T* b,
                        std::int64_t ldb,
                        const std::vector<sycl::event> &dependencies = {})
   }

.. container:: section

   .. rubric:: Input Parameters

   queue
      The queue where the routine should be executed.

   left_right
      Specifies whether ``A`` multiplies ``X`` on the left
      (``side::left``) or on the right (``side::right``). See :ref:`onemkl_datatypes` for more details.

   uplo
      Specifies whether the matrix ``A`` is upper or lower
      triangular. See :ref:`onemkl_datatypes` for more details.

   transa
      Specifies op(``A``), the transposition operation applied to
      ``A``. See :ref:`onemkl_datatypes` for more details.

   unit_diag
      Specifies whether ``A`` is assumed to be unit triangular (all
      diagonal elements are 1). See :ref:`onemkl_datatypes` for more details.

   m
      Specifies the number of rows of ``B``. The value of ``m`` must
      be at least zero.

   n
      Specifies the number of columns of ``B``. The value of ``n``
      must be at least zero.

   alpha
      Scaling factor for the solution.

   a
      Pointer to input matrix ``A``. Must have size at least
      ``lda``\ \*\ ``m`` if ``left_right`` = ``side::left``, or
      ``lda``\ \*\ ``n`` if ``left_right`` = ``side::right``. See
      :ref:`matrix-storage` for
      more details.

   lda
      Leading dimension of ``A``. Must be at least ``m`` if
      ``left_right`` = ``side::left``, and at least ``n`` if
      ``left_right`` = ``side::right``. Must be positive.

   b
      Pointer to input/output matrix ``B``. Must have size at
      least ``ldb``\ \*\ ``n`` if column major layout is used to store
      matrices or at least ``ldb``\ \*\ ``m`` if row major layout is
      used to store matrices. See :ref:`matrix-storage` for more details.

   ldb
      Leading dimension of ``B``. It must be positive and at least
      ``m`` if column major layout is used to store matrices or at
      least ``n`` if row major layout is used to store matrices.

   dependencies
      List of events to wait for before starting computation, if any.
      If omitted, defaults to no dependencies.

.. container:: section

   .. rubric:: Output Parameters

   b
      Pointer to the output matrix. Overwritten by the solution
      matrix ``X``.

.. container:: section

   .. rubric:: Notes

   If ``alpha`` = 0, matrix ``B`` is set to zero, and ``A`` and ``B``
   do not need to be initialized at entry.

.. container:: section

   .. rubric:: Return Values

   Output event to wait on to ensure computation is complete.


   **Parent topic:** :ref:`blas-level-3-routines`
